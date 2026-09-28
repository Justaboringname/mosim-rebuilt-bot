// Whole-match state capture / restore, so another instance of the same build can continue a match from exactly
// where this one is (lookahead planning: the main match only ever snapshots; planner instances restore and roll out).
//
// Identity: objects are keyed by their hierarchy path at episode start (name + index among same-named siblings), so
// keys agree across instances even though FindObjectsByType order and RebuiltGamePieceManager slot numbers do not.
// State captured:
//   - every Rigidbody: pose, velocity, flags, sleep, active;   - every ConfigurableJoint: motions, targets, drives;
//   - every game MonoBehaviour's fields, generically by reflection (value fields, structs, plain objects in place,
//     arrays / lists / sets, references to scene objects by key; delegates, native buffers, input actions skipped);
//   - RebuiltGamePieceManager's per-slot arrays, re-indexed by ball key;
//   - the mutable statics of the REBUILT game (shift state, scores, zone penalty);   - UnityEngine.Random.state.
// Stored Time.time stamps are shifted by the clock difference between the two instances.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Games.Rebuilt.GamePieceSystem;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace MoSimRL
{
    public static class Snapshot
    {
        private const int Version = 2;     // 2: gzip-compressed body
        private const BindingFlags Inst = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        private const BindingFlags Stat = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;

        private static readonly HashSet<string> GameAssemblies = new HashSet<string>
            { "Rebuilt", "RobotFramework", "GameSystems", "MoSimCore", "MoSimLib", "Assembly-CSharp" };

        private static readonly string[] SkipTypeWords =
            { "Audio", "Camera", "Discord", "Menu", "Tutorial", "Steam", "Leaderboard", "Settings", "Transition", "Presence", "Replay" };

        // fields that store a Time.time stamp (game fields assigned from Time.time, version 26.4.1)
        private static readonly HashSet<string> TimeFields = new HashSet<string>
        {
            "Games.Rebuilt.GamePieceSystem.RebuiltGamePieceController._lastScoredTime",
            "Games.Rebuilt.GamePieceSystem.RebuiltGamePieceController.lastZonePenaltyTime",
            "RobotFramework.Components.GenericJoint._lastTime",
            "RobotFramework.Components.InteractionRoller._lastTime",
        };

        // per-process identities that must stay as they are in the target instance
        private static readonly HashSet<string> SkipFields = new HashSet<string>
        {
            "Games.Rebuilt.GamePieceSystem.RebuiltGamePieceController.SlotIndex",
        };

        private static readonly string[] StaticTypes =
        {
            "Games.Rebuilt.FieldScripts.RebuiltShifts", "Games.Rebuilt.Scoring.RebuiltScoreUI", "Games.Rebuilt.Scoring.ZoneShotsPenalty",
            "Games.Rebuilt.GamePieceSystem.RebuiltGamePieceController", "Games.Rebuilt.Scoring.HubScoring",
        };

        private enum Tag : byte { Null, Bool, Int, Float, Double, Enum, Vec, Struct, Array, URef, Obj, Skip }

        // ------------------------------------------------------------------ identity

        private static readonly Dictionary<GameObject, string> GoKey = new Dictionary<GameObject, string>();
        private static readonly Dictionary<string, GameObject> KeyGo = new Dictionary<string, GameObject>();
        public static int DuplicateKeys { get; private set; }
        public static int KeyCount => KeyGo.Count;

        public static void BuildKeys()
        {
            GoKey.Clear(); KeyGo.Clear(); DuplicateKeys = 0;
            for (int s = 0; s < SceneManager.sceneCount; s++)
            {
                var sc = SceneManager.GetSceneAt(s);
                if (!sc.isLoaded) continue;
                var seen = new Dictionary<string, int>();
                foreach (var r in sc.GetRootGameObjects())
                {
                    if (r.GetComponentInChildren<Bridge>(true) != null) continue;      // our own host object
                    seen.TryGetValue(r.name, out int k); seen[r.name] = k + 1;
                    Walk(r.transform, sc.name + ":" + r.name + "#" + k);
                }
            }
        }

        private static void Walk(Transform t, string key)
        {
            if (KeyGo.ContainsKey(key)) DuplicateKeys++;
            else { KeyGo[key] = t.gameObject; GoKey[t.gameObject] = key; }
            var seen = new Dictionary<string, int>();
            for (int i = 0; i < t.childCount; i++)
            {
                var c = t.GetChild(i);
                seen.TryGetValue(c.name, out int k); seen[c.name] = k + 1;
                Walk(c, key + "/" + c.name + "#" + k);
            }
        }

        public static string KeyOf(GameObject go) => go != null && GoKey.TryGetValue(go, out var k) ? k : null;

        private static string RefKey(UnityEngine.Object o)
        {
            if (o == null) return "";
            if (o is GameObject go) return KeyOf(go) ?? "";
            if (o is Component c)
            {
                string gk = KeyOf(c.gameObject);
                if (gk == null) return "";
                var t = c.GetType();
                int idx = 0;
                foreach (var o2 in c.gameObject.GetComponents<Component>())      // index among components of exactly this type
                {
                    if (ReferenceEquals(o2, c)) break;
                    if (o2 != null && o2.GetType() == t) idx++;
                }
                return gk + "|" + t.FullName + "#" + idx;
            }
            return "";                                                               // assets: never restored
        }

        private static UnityEngine.Object Resolve(string key, Type want)
        {
            if (string.IsNullOrEmpty(key)) return null;
            int bar = key.IndexOf('|');
            if (bar < 0) return KeyGo.TryGetValue(key, out var go) && want.IsAssignableFrom(typeof(GameObject)) ? go : null;
            if (!KeyGo.TryGetValue(key.Substring(0, bar), out var g) || g == null) return null;
            string rest = key.Substring(bar + 1);
            int hash = rest.LastIndexOf('#');
            string tname = rest.Substring(0, hash);
            int idx = int.Parse(rest.Substring(hash + 1));
            foreach (var c in g.GetComponents<Component>())
                if (c != null && c.GetType().FullName == tname)
                {
                    if (idx == 0) return want.IsInstanceOfType(c) ? c : null;
                    idx--;
                }
            return null;
        }

        // ------------------------------------------------------------------ what gets captured

        private static bool SkipType(Type t)
        {
            if (t.Namespace != null && (t.Namespace.StartsWith("MoSimRL") || t.Namespace.StartsWith("TMPro") || t.Namespace.StartsWith("UnityEngine")))
                return true;
            if (!GameAssemblies.Contains(t.Assembly.GetName().Name)) return true;
            if (t == typeof(RebuiltGamePieceManager)) return true;                     // per-slot arrays handled by key
            foreach (var w in SkipTypeWords) if (t.Name.Contains(w)) return true;
            return false;
        }

        private static List<MonoBehaviour> Components()
        {
            var list = new List<(string, MonoBehaviour)>();
            foreach (var mb in UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (mb == null || SkipType(mb.GetType())) continue;
                string k = RefKey(mb);
                if (k == "") continue;
                list.Add((k, mb));
            }
            list.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
            return list.Select(p => p.Item2).ToList();
        }

        private static readonly Dictionary<Type, FieldInfo[]> FieldCache = new Dictionary<Type, FieldInfo[]>();

        private static FieldInfo[] Fields(Type t)
        {
            if (FieldCache.TryGetValue(t, out var fs)) return fs;
            var all = new List<FieldInfo>();
            for (var tt = t; tt != null && tt != typeof(MonoBehaviour) && tt != typeof(object) && tt != typeof(ValueType); tt = tt.BaseType)
                foreach (var f in tt.GetFields(Inst))
                    if (!f.IsLiteral && !typeof(Delegate).IsAssignableFrom(f.FieldType) && !SkipFields.Contains(FieldId(f))) all.Add(f);
            all.Sort((a, b) => string.CompareOrdinal(FieldId(a), FieldId(b)));
            FieldCache[t] = fs = all.ToArray();
            return fs;
        }

        private static readonly Dictionary<Type, Dictionary<string, FieldInfo>> ByIdCache = new Dictionary<Type, Dictionary<string, FieldInfo>>();

        private static string FieldId(FieldInfo f) => f.DeclaringType.FullName + "." + f.Name;

        private static bool Unsupported(Type t)
        {
            if (t == typeof(string) || t == typeof(IntPtr) || t == typeof(UIntPtr)) return true;
            if (typeof(Delegate).IsAssignableFrom(t)) return true;
            string ns = t.Namespace ?? "";
            if (ns.StartsWith("Unity.Collections") || ns.StartsWith("Unity.Jobs") || ns.StartsWith("UnityEngine.InputSystem")) return true;
            if (t.Name == "MoSimInput") return true;
            if (!t.IsValueType && !typeof(UnityEngine.Object).IsAssignableFrom(t) && ns.StartsWith("UnityEngine")) return true;   // Coroutine, AnimationCurve...
            if (t.IsPointer) return true;
            if (t.IsGenericType)
            {
                var g = t.GetGenericTypeDefinition();
                if (g == typeof(Dictionary<,>) || g == typeof(Queue<>) || g == typeof(Stack<>)) return true;
            }
            return false;
        }

        private static bool IsVec(Type t) => t == typeof(Vector2) || t == typeof(Vector3) || t == typeof(Vector4) || t == typeof(Quaternion) || t == typeof(Color);

        // ------------------------------------------------------------------ generic writer

        private sealed class Ctx
        {
            public readonly HashSet<object> Visited = new HashSet<object>(RefEq.I);
            public int Skipped;
        }

        private sealed class RefEq : IEqualityComparer<object>
        {
            public static readonly RefEq I = new RefEq();
            public new bool Equals(object a, object b) => ReferenceEquals(a, b);
            public int GetHashCode(object o) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(o);
        }

        private static void WriteFields(BinaryWriter w, object obj, Type t, int depth, Ctx ctx)
        {
            var fs = Fields(t);
            w.Write(fs.Length);
            foreach (var f in fs)
            {
                w.Write(FieldId(f));
                object v;
                try { v = f.GetValue(obj); } catch { v = null; w.Write((byte)Tag.Skip); ctx.Skipped++; continue; }
                WriteValue(w, v, f.FieldType, depth, ctx);
            }
        }

        private static void WriteValue(BinaryWriter w, object v, Type declared, int depth, Ctx ctx)
        {
            if (v == null) { w.Write((byte)Tag.Null); return; }
            var t = v.GetType();
            if (Unsupported(t)) { w.Write((byte)Tag.Skip); ctx.Skipped++; return; }
            if (t == typeof(bool)) { w.Write((byte)Tag.Bool); w.Write((bool)v); return; }
            if (t.IsEnum) { w.Write((byte)Tag.Enum); w.Write(Enum.GetUnderlyingType(t) == typeof(ulong) ? unchecked((long)System.Convert.ToUInt64(v)) : System.Convert.ToInt64(v)); return; }
            if (t == typeof(float)) { w.Write((byte)Tag.Float); w.Write((float)v); return; }
            if (t == typeof(double)) { w.Write((byte)Tag.Double); w.Write((double)v); return; }
            if (t == typeof(ulong)) { w.Write((byte)Tag.Int); w.Write(unchecked((long)(ulong)v)); return; }
            if (t.IsPrimitive) { w.Write((byte)Tag.Int); w.Write(System.Convert.ToInt64(v)); return; }
            if (IsVec(t))
            {
                w.Write((byte)Tag.Vec);
                float[] a = t == typeof(Vector2) ? new[] { ((Vector2)v).x, ((Vector2)v).y }
                    : t == typeof(Vector3) ? new[] { ((Vector3)v).x, ((Vector3)v).y, ((Vector3)v).z }
                    : t == typeof(Vector4) ? new[] { ((Vector4)v).x, ((Vector4)v).y, ((Vector4)v).z, ((Vector4)v).w }
                    : t == typeof(Quaternion) ? new[] { ((Quaternion)v).x, ((Quaternion)v).y, ((Quaternion)v).z, ((Quaternion)v).w }
                    : new[] { ((Color)v).r, ((Color)v).g, ((Color)v).b, ((Color)v).a };
                w.Write((byte)a.Length);
                foreach (var x in a) w.Write(x);
                return;
            }
            if (v is UnityEngine.Object uo) { w.Write((byte)Tag.URef); w.Write(RefKey(uo)); return; }
            if (depth > 5) { w.Write((byte)Tag.Skip); ctx.Skipped++; return; }
            if (t.IsArray || (t.IsGenericType && (t.GetGenericTypeDefinition() == typeof(List<>) || t.GetGenericTypeDefinition() == typeof(HashSet<>))))
            {
                var et = t.IsArray ? t.GetElementType() : t.GetGenericArguments()[0];
                if (Unsupported(et)) { w.Write((byte)Tag.Skip); ctx.Skipped++; return; }   // e.g. Scheduler's List<Action>
                var items = new List<object>();
                foreach (var x in (IEnumerable)v) items.Add(x);
                if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(HashSet<>))
                    items.Sort((a, b) => string.CompareOrdinal(ElemSortKey(a), ElemSortKey(b)));   // set order is per-process
                w.Write((byte)Tag.Array);
                w.Write(items.Count);
                foreach (var x in items) WriteValue(w, x, typeof(object), depth + 1, ctx);
                return;
            }
            if (t.IsValueType) { w.Write((byte)Tag.Struct); WriteFields(w, v, t, depth + 1, ctx); return; }
            if (ctx.Visited.Contains(v)) { w.Write((byte)Tag.Skip); return; }
            ctx.Visited.Add(v);
            w.Write((byte)Tag.Obj);
            WriteFields(w, v, t, depth + 1, ctx);
        }

        private static string ElemSortKey(object o) => o is UnityEngine.Object u ? RefKey(u) : System.Convert.ToString(o, System.Globalization.CultureInfo.InvariantCulture);

        // ------------------------------------------------------------------ generic reader

        // Parsed value: primitives boxed, Vec as float[], Struct/Obj as field list, Array as list, URef as string.
        private sealed class PVal { public Tag Tag; public object V; }

        private static List<(string, PVal)> ReadFields(BinaryReader r)
        {
            int n = r.ReadInt32();
            var list = new List<(string, PVal)>(n);
            for (int i = 0; i < n; i++) list.Add((r.ReadString(), ReadValue(r)));
            return list;
        }

        private static PVal ReadValue(BinaryReader r)
        {
            var tag = (Tag)r.ReadByte();
            switch (tag)
            {
                case Tag.Null: case Tag.Skip: return new PVal { Tag = tag };
                case Tag.Bool: return new PVal { Tag = tag, V = r.ReadBoolean() };
                case Tag.Int: case Tag.Enum: return new PVal { Tag = tag, V = r.ReadInt64() };
                case Tag.Float: return new PVal { Tag = tag, V = r.ReadSingle() };
                case Tag.Double: return new PVal { Tag = tag, V = r.ReadDouble() };
                case Tag.Vec: { int n = r.ReadByte(); var a = new float[n]; for (int i = 0; i < n; i++) a[i] = r.ReadSingle(); return new PVal { Tag = tag, V = a }; }
                case Tag.URef: return new PVal { Tag = tag, V = r.ReadString() };
                case Tag.Struct: case Tag.Obj: return new PVal { Tag = tag, V = ReadFields(r) };
                case Tag.Array: { int n = r.ReadInt32(); var l = new List<PVal>(n); for (int i = 0; i < n; i++) l.Add(ReadValue(r)); return new PVal { Tag = tag, V = l }; }
            }
            throw new InvalidDataException("bad tag " + (int)tag);
        }

        private sealed class Apply
        {
            public float TimeShift;
            public int Set, Missing, Unresolved, Failed;
        }

        private static void ApplyFields(object obj, Type t, List<(string, PVal)> vals, Apply ap)
        {
            if (!ByIdCache.TryGetValue(t, out var byId)) ByIdCache[t] = byId = Fields(t).ToDictionary(FieldId);
            foreach (var (id, pv) in vals)
            {
                if (pv.Tag == Tag.Skip) continue;
                if (!byId.TryGetValue(id, out var f)) { ap.Missing++; continue; }
                try
                {
                    object cur = f.GetValue(obj);
                    bool changed;
                    object nv = ConvertVal(pv, f.FieldType, cur, TimeFields.Contains(id), ap, out changed);
                    if (changed) { f.SetValue(obj, nv); ap.Set++; }
                }
                catch (Exception) { ap.Failed++; }
            }
        }

        // Returns the value to store; `changed` false means leave the field alone (in-place update or nothing to do).
        private static object ConvertVal(PVal pv, Type ft, object cur, bool isTime, Apply ap, out bool changed)
        {
            changed = true;
            var under = Nullable.GetUnderlyingType(ft);
            var tt = under ?? ft;
            switch (pv.Tag)
            {
                case Tag.Null:
                    if (under != null || typeof(UnityEngine.Object).IsAssignableFrom(ft)) return null;
                    changed = false; return null;
                case Tag.Bool: return (bool)pv.V;
                case Tag.Float:
                {
                    float x = (float)pv.V;
                    if (isTime && !float.IsInfinity(x)) x += ap.TimeShift;
                    return tt == typeof(float) ? (object)x : System.Convert.ChangeType(x, tt);
                }
                case Tag.Double: return tt == typeof(double) ? pv.V : System.Convert.ChangeType(pv.V, tt);
                case Tag.Int: return tt == typeof(ulong) ? unchecked((ulong)(long)pv.V) : System.Convert.ChangeType(pv.V, tt);
                case Tag.Enum: return tt.IsEnum ? Enum.ToObject(tt, (long)pv.V) : System.Convert.ChangeType(pv.V, tt);
                case Tag.Vec:
                {
                    var a = (float[])pv.V;
                    if (tt == typeof(Vector2)) return new Vector2(a[0], a[1]);
                    if (tt == typeof(Vector3)) return new Vector3(a[0], a[1], a[2]);
                    if (tt == typeof(Vector4)) return new Vector4(a[0], a[1], a[2], a[3]);
                    if (tt == typeof(Quaternion)) return new Quaternion(a[0], a[1], a[2], a[3]);
                    if (tt == typeof(Color)) return new Color(a[0], a[1], a[2], a[3]);
                    changed = false; return null;
                }
                case Tag.URef:
                {
                    var o = Resolve((string)pv.V, ft);
                    if (o == null) { if (!string.IsNullOrEmpty((string)pv.V)) ap.Unresolved++; changed = false; return null; }
                    if (ReferenceEquals(o, cur)) { changed = false; return null; }
                    return o;
                }
                case Tag.Struct:
                {
                    object box = cur ?? Activator.CreateInstance(tt);
                    ApplyFields(box, tt, (List<(string, PVal)>)pv.V, ap);
                    return box;
                }
                case Tag.Obj:
                    changed = false;
                    if (cur != null) ApplyFields(cur, cur.GetType(), (List<(string, PVal)>)pv.V, ap);
                    return null;
                case Tag.Array:
                    return ConvertArray((List<PVal>)pv.V, ft, cur, ap, out changed);
            }
            changed = false; return null;
        }

        private static object ConvertArray(List<PVal> items, Type ft, object cur, Apply ap, out bool changed)
        {
            changed = false;
            if (ft.IsArray)
            {
                var et = ft.GetElementType();
                var arr = cur as Array;
                if (arr == null || arr.Length != items.Count) { arr = Array.CreateInstance(et, items.Count); changed = true; }
                for (int i = 0; i < items.Count; i++)
                {
                    object old = arr.GetValue(i);
                    if (items[i].Tag == Tag.Obj && old == null) continue;
                    object nv = ConvertVal(items[i], et, old, false, ap, out bool ch);
                    if (ch) arr.SetValue(nv, i);
                }
                return arr;
            }
            if (!ft.IsGenericType || cur == null) return null;
            var g = ft.GetGenericTypeDefinition();
            var elt = ft.GetGenericArguments()[0];
            bool plain = !elt.IsValueType && !typeof(UnityEngine.Object).IsAssignableFrom(elt);
            if (g == typeof(List<>))
            {
                var list = (IList)cur;
                if (plain)
                {
                    // lists of plain objects: update the existing elements in place, never rebuild
                    if (list.Count != items.Count) { ap.Failed++; return null; }
                    for (int i = 0; i < items.Count; i++)
                        if (items[i].Tag == Tag.Obj && list[i] != null) ApplyFields(list[i], list[i].GetType(), (List<(string, PVal)>)items[i].V, ap);
                    return null;
                }
                var fresh = new List<object>(items.Count);
                foreach (var it in items)
                {
                    object nv = ConvertVal(it, elt, null, false, ap, out bool ch);
                    if (!ch && it.Tag != Tag.Null) { ap.Failed++; return null; }      // unresolved element: leave the list alone
                    fresh.Add(nv);
                }
                list.Clear();
                foreach (var x in fresh) list.Add(x);
                return null;
            }
            if (g == typeof(HashSet<>))
            {
                if (plain) return null;
                var fresh = new List<object>(items.Count);
                foreach (var it in items)
                {
                    object nv = ConvertVal(it, elt, null, false, ap, out bool ch);
                    if (!ch || nv == null) { ap.Failed++; return null; }
                    fresh.Add(nv);
                }
                ft.GetMethod("Clear").Invoke(cur, null);
                var add = ft.GetMethod("Add");
                foreach (var x in fresh) add.Invoke(cur, new[] { x });
            }
            return null;
        }

        private sealed class SlotArrays
        {
            public readonly Vector3[] ReturnPoints; public readonly bool[] HasReturnPoint, WatchingFor404, Scored, HasOwner;
            public readonly float[] ReturnCounters; public readonly Array OwnedBy;
            public SlotArrays(RebuiltGamePieceManager m)
            {
                const BindingFlags bf = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
                var t = m.GetType();
                object F(string n) => t.GetField(n, bf).GetValue(m);
                ReturnPoints = (Vector3[])F("ReturnPoints"); HasReturnPoint = (bool[])F("HasReturnPoint");
                ReturnCounters = (float[])F("ReturnCounters"); WatchingFor404 = (bool[])F("WatchingFor404");
                Scored = (bool[])F("Scored"); OwnedBy = (Array)F("OwnedBy"); HasOwner = (bool[])F("HasOwner");
            }
        }

        // ------------------------------------------------------------------ capture

        public static byte[] Capture(Bridge.PadState pad, out string summary)
        {
            var ms = new MemoryStream(1 << 18);
            var w = new BinaryWriter(ms);
            var ctx = new Ctx();
            w.Write(Version);
            w.Write(Time.time);
            w.Write(JsonUtility.ToJson(UnityEngine.Random.state));
            pad.Write(w);

            // statics
            var stypes = StaticTypes.Select(FindType).Where(x => x != null).ToList();
            w.Write(stypes.Count);
            foreach (var t in stypes)
            {
                w.Write(t.FullName);
                var fs = t.GetFields(Stat).Where(f => !f.IsLiteral && !f.IsInitOnly && (f.FieldType.IsValueType)).OrderBy(f => f.Name, StringComparer.Ordinal).ToArray();
                w.Write(fs.Length);
                foreach (var f in fs) { w.Write(FieldId(f)); WriteValue(w, f.GetValue(null), f.FieldType, 0, ctx); }
            }

            // components
            var comps = Components();
            w.Write(comps.Count);
            foreach (var c in comps) { w.Write(RefKey(c)); WriteFields(w, c, c.GetType(), 0, ctx); }

            // RebuiltGamePieceManager per-slot arrays (internal fields), keyed by ball
            var mgr = RebuiltGamePieceManager.Instance;
            var sa = mgr != null ? new SlotArrays(mgr) : null;
            var balls = UnityEngine.Object.FindObjectsByType<RebuiltGamePieceController>(FindObjectsInactive.Include, FindObjectsSortMode.None)
                .Where(b => KeyOf(b.gameObject) != null).OrderBy(b => KeyOf(b.gameObject), StringComparer.Ordinal).ToList();
            w.Write(mgr != null ? balls.Count : 0);
            if (mgr != null)
                foreach (var b in balls)
                {
                    int s = b.SlotIndex;
                    w.Write(KeyOf(b.gameObject));
                    bool ok = s >= 0 && s < sa.Scored.Length;
                    w.Write(ok);
                    if (!ok) continue;
                    var rp = sa.ReturnPoints[s];
                    w.Write(rp.x); w.Write(rp.y); w.Write(rp.z);
                    w.Write(sa.HasReturnPoint[s]); w.Write(sa.ReturnCounters[s]); w.Write(sa.WatchingFor404[s]);
                    w.Write(sa.Scored[s]); w.Write(System.Convert.ToInt32(sa.OwnedBy.GetValue(s))); w.Write(sa.HasOwner[s]);
                }

            // joints
            var joints = UnityEngine.Object.FindObjectsByType<ConfigurableJoint>(FindObjectsInactive.Include, FindObjectsSortMode.None)
                .Select(j => (k: RefKey(j), j)).Where(p => p.k != "").OrderBy(p => p.k, StringComparer.Ordinal).ToList();
            w.Write(joints.Count);
            foreach (var (k, j) in joints)
            {
                w.Write(k);
                w.Write((int)j.xMotion); w.Write((int)j.yMotion); w.Write((int)j.zMotion);
                w.Write((int)j.angularXMotion); w.Write((int)j.angularYMotion); w.Write((int)j.angularZMotion);
                WV(w, j.targetPosition); WQ(w, j.targetRotation); WV(w, j.targetVelocity); WV(w, j.targetAngularVelocity);
                WD(w, j.xDrive); WD(w, j.yDrive); WD(w, j.zDrive); WD(w, j.angularXDrive); WD(w, j.angularYZDrive); WD(w, j.slerpDrive);
            }

            // rigidbodies, parents before children
            var rbs = UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include, FindObjectsSortMode.None)
                .Select(b => (k: KeyOf(b.gameObject), b)).Where(p => p.k != null)
                .OrderBy(p => Depth(p.b.transform)).ThenBy(p => p.k, StringComparer.Ordinal).ToList();
            w.Write(rbs.Count);
            foreach (var (k, b) in rbs)
            {
                w.Write(k);
                w.Write(b.gameObject.activeSelf);
                w.Write(KeyOf(b.transform.parent != null ? b.transform.parent.gameObject : null) ?? "");
                WV(w, b.position); WQ(w, b.rotation); WV(w, b.velocity); WV(w, b.angularVelocity);
                w.Write(b.isKinematic); w.Write(b.useGravity); w.Write(b.detectCollisions); w.Write(b.IsSleeping());
                w.Write((int)b.excludeLayers); w.Write(b.mass); w.Write(b.drag); w.Write(b.angularDrag); w.Write((int)b.constraints);
            }
            w.Flush();
            // ~4 MB raw, mostly repeated field ids and hierarchy keys: gzip makes the transfer to planners cheap
            var zs = new MemoryStream((int)(ms.Length / 6));
            using (var gz = new System.IO.Compression.GZipStream(zs, System.IO.Compression.CompressionLevel.Fastest, true))
                gz.Write(ms.GetBuffer(), 0, (int)ms.Length);
            summary = $"statics={stypes.Count} comps={comps.Count} balls={balls.Count} joints={joints.Count} rbs={rbs.Count} skipped={ctx.Skipped} bytes={ms.Length} gz={zs.Length}";
            return zs.ToArray();
        }

        private static int Depth(Transform t) { int d = 0; while (t.parent != null) { d++; t = t.parent; } return d; }
        private static void WV(BinaryWriter w, Vector3 v) { w.Write(v.x); w.Write(v.y); w.Write(v.z); }
        private static void WQ(BinaryWriter w, Quaternion q) { w.Write(q.x); w.Write(q.y); w.Write(q.z); w.Write(q.w); }
        private static void WD(BinaryWriter w, JointDrive d) { w.Write(d.positionSpring); w.Write(d.positionDamper); w.Write(d.maximumForce); }
        private static Vector3 RV(BinaryReader r) => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
        private static Quaternion RQ(BinaryReader r) => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
        private static JointDrive RD(BinaryReader r) => new JointDrive { positionSpring = r.ReadSingle(), positionDamper = r.ReadSingle(), maximumForce = r.ReadSingle() };

        private static Type FindType(string name)
        {
            foreach (var a in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = a.GetType(name, false);
                if (t != null) return t;
            }
            return null;
        }

        // ------------------------------------------------------------------ restore

        public static string Restore(byte[] blob, out Bridge.PadState pad)
        {
            var raw = new MemoryStream(blob.Length * 8);
            using (var gz = new System.IO.Compression.GZipStream(new MemoryStream(blob), System.IO.Compression.CompressionMode.Decompress))
                gz.CopyTo(raw);
            raw.Position = 0;
            var r = new BinaryReader(raw);
            int ver = r.ReadInt32();
            if (ver != Version) throw new InvalidDataException($"snapshot version {ver} != {Version}");
            var ap = new Apply();
            float srcTime = r.ReadSingle();
            ap.TimeShift = Time.time - srcTime;
            var rnd = JsonUtility.FromJson<UnityEngine.Random.State>(r.ReadString());
            pad = Bridge.PadState.Read(r);

            int nst = r.ReadInt32();
            for (int i = 0; i < nst; i++)
            {
                var t = FindType(r.ReadString());
                int nf = r.ReadInt32();
                for (int j = 0; j < nf; j++)
                {
                    string id = r.ReadString();
                    var pv = ReadValue(r);
                    var f = t?.GetFields(Stat).FirstOrDefault(x => FieldId(x) == id);
                    if (f == null) { ap.Missing++; continue; }
                    try
                    {
                        object nv = ConvertVal(pv, f.FieldType, f.GetValue(null), TimeFields.Contains(id), ap, out bool ch);
                        if (ch) { f.SetValue(null, nv); ap.Set++; }
                    }
                    catch { ap.Failed++; }
                }
            }

            var comps = Components().ToDictionary(c => RefKey(c));
            int nc = r.ReadInt32(), cMissing = 0;
            for (int i = 0; i < nc; i++)
            {
                string k = r.ReadString();
                var vals = ReadFields(r);
                if (!comps.TryGetValue(k, out var c)) { cMissing++; continue; }
                ApplyFields(c, c.GetType(), vals, ap);
            }

            var mgr = RebuiltGamePieceManager.Instance;
            var sa = mgr != null ? new SlotArrays(mgr) : null;
            var ballByKey = new Dictionary<string, RebuiltGamePieceController>();
            foreach (var b in UnityEngine.Object.FindObjectsByType<RebuiltGamePieceController>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                var bk = KeyOf(b.gameObject);
                if (bk != null) ballByKey[bk] = b;
            }
            int nb = r.ReadInt32(), bMissing = 0;
            for (int i = 0; i < nb; i++)
            {
                string k = r.ReadString();
                if (!r.ReadBoolean()) continue;
                var rp = RV(r);
                bool hrp = r.ReadBoolean(); float rc = r.ReadSingle(); bool w404 = r.ReadBoolean();
                bool sc = r.ReadBoolean(); int own = r.ReadInt32(); bool hown = r.ReadBoolean();
                if (sa == null || !ballByKey.TryGetValue(k, out var b) || b.SlotIndex < 0 || b.SlotIndex >= sa.Scored.Length) { bMissing++; continue; }
                int s = b.SlotIndex;
                sa.ReturnPoints[s] = rp; sa.HasReturnPoint[s] = hrp; sa.ReturnCounters[s] = rc; sa.WatchingFor404[s] = w404;
                sa.Scored[s] = sc; sa.OwnedBy.SetValue(Enum.ToObject(sa.OwnedBy.GetType().GetElementType(), own), s); sa.HasOwner[s] = hown;
            }

            int nj = r.ReadInt32(), jMissing = 0;
            for (int i = 0; i < nj; i++)
            {
                string k = r.ReadString();
                var m = new int[6]; for (int q = 0; q < 6; q++) m[q] = r.ReadInt32();
                var tp = RV(r); var tr = RQ(r); var tv = RV(r); var tav = RV(r);
                var d = new JointDrive[6]; for (int q = 0; q < 6; q++) d[q] = RD(r);
                var j = Resolve(k, typeof(ConfigurableJoint)) as ConfigurableJoint;
                if (j == null) { jMissing++; continue; }
                if ((int)j.xMotion != m[0]) j.xMotion = (ConfigurableJointMotion)m[0];
                if ((int)j.yMotion != m[1]) j.yMotion = (ConfigurableJointMotion)m[1];
                if ((int)j.zMotion != m[2]) j.zMotion = (ConfigurableJointMotion)m[2];
                if ((int)j.angularXMotion != m[3]) j.angularXMotion = (ConfigurableJointMotion)m[3];
                if ((int)j.angularYMotion != m[4]) j.angularYMotion = (ConfigurableJointMotion)m[4];
                if ((int)j.angularZMotion != m[5]) j.angularZMotion = (ConfigurableJointMotion)m[5];
                j.targetPosition = tp; j.targetRotation = tr; j.targetVelocity = tv; j.targetAngularVelocity = tav;
                j.xDrive = d[0]; j.yDrive = d[1]; j.zDrive = d[2]; j.angularXDrive = d[3]; j.angularYZDrive = d[4]; j.slerpDrive = d[5];
            }

            int nr = r.ReadInt32(), rMissing = 0, reparented = 0;
            var sleepers = new List<Rigidbody>();
            for (int i = 0; i < nr; i++)
            {
                string k = r.ReadString();
                bool active = r.ReadBoolean();
                string parent = r.ReadString();
                var pos = RV(r); var rot = RQ(r); var vel = RV(r); var avel = RV(r);
                bool kin = r.ReadBoolean(), grav = r.ReadBoolean(), det = r.ReadBoolean(), sleeping = r.ReadBoolean();
                int excl = r.ReadInt32(); float mass = r.ReadSingle(), drag = r.ReadSingle(), adrag = r.ReadSingle(); int cons = r.ReadInt32();
                if (!KeyGo.TryGetValue(k, out var go) || go == null) { rMissing++; continue; }
                var b = go.GetComponent<Rigidbody>();
                if (b == null) { rMissing++; continue; }
                if (go.activeSelf != active) go.SetActive(active);
                string curParent = KeyOf(go.transform.parent != null ? go.transform.parent.gameObject : null) ?? "";
                if (curParent != parent)
                {
                    reparented++;
                    var pgo = parent == "" ? null : (KeyGo.TryGetValue(parent, out var pg) ? pg : null);
                    if (parent == "" || pgo != null) go.transform.SetParent(pgo != null ? pgo.transform : null, true);
                }
                if (b.isKinematic != kin) b.isKinematic = kin;
                b.useGravity = grav; b.detectCollisions = det; b.excludeLayers = excl;
                b.mass = mass; b.drag = drag; b.angularDrag = adrag; b.constraints = (RigidbodyConstraints)cons;
                go.transform.SetPositionAndRotation(pos, rot);
                b.position = pos; b.rotation = rot;
                if (!kin) { b.velocity = vel; b.angularVelocity = avel; }
                if (sleeping) sleepers.Add(b); else b.WakeUp();
            }
            Physics.SyncTransforms();
            // GenericJoint.lockAllAxis re-anchors the joint at the pose it has when it locks (toggling
            // autoConfigureConnectedAnchor makes Unity rebuild the joint around the current relative pose). A restored
            // lock must be re-anchored the same way at the restored pose, or the locked part (4414's kicker bar) is
            // held at whatever pose this instance's joint was built around.
            int reanchored = 0;
            var gjType = FindType("RobotFramework.Components.GenericJoint");
            var wasLockedF = gjType?.GetField("wasLocked", BindingFlags.Instance | BindingFlags.NonPublic);
            if (gjType != null && wasLockedF != null)
                foreach (var gj in UnityEngine.Object.FindObjectsByType(gjType, FindObjectsInactive.Include, FindObjectsSortMode.None))
                {
                    var comp = gj as Component;
                    if (comp == null || KeyOf(comp.gameObject) == null || !(bool)wasLockedF.GetValue(gj)) continue;
                    var cj = comp.GetComponent<ConfigurableJoint>();
                    if (cj == null) continue;
                    cj.autoConfigureConnectedAnchor = false;
                    cj.autoConfigureConnectedAnchor = true;
                    reanchored++;
                }
            foreach (var b in sleepers) b.Sleep();
            UnityEngine.Random.state = rnd;

            return $"set={ap.Set} missingFields={ap.Missing} unresolved={ap.Unresolved} failed={ap.Failed} compsMissing={cMissing} " +
                   $"ballsMissing={bMissing} jointsMissing={jMissing} rbsMissing={rMissing} reparented={reparented} reanchored={reanchored} timeShift={ap.TimeShift:F3}";
        }
    }
}
