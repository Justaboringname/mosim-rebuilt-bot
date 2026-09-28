// Records a HUMAN driving 4414 (run/RECORD): every 0.1 s of game time, the robot state, match state and the
// driver's raw inputs, one JSONL file per match under run/demos/. Used to learn the human's strategy
// (routes, collection spots, shooting spots, timing) and later for behaviour cloning / ghost-tracking rewards.
// Recording never touches the controls and never blocks leaderboard submission.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Games.Rebuilt.FieldScripts;
using Games.Rebuilt.GamePieceSystem;
using Games.Rebuilt.Robots;
using Games.Rebuilt.Scoring;
using GameSystems.Management;
using MoSimCore.BaseClasses.GameManagement;
using UnityEngine;

namespace MoSimRL
{
    public class Recorder : MonoBehaviour
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        private const int StepsPerRecord = 22;          // 22 x 4.5 ms ≈ 0.1 s
        private const int FuelEvery = 5;                // full fuel snapshot every 0.5 s

        private GameObject _robot;
        private RebuiltRobotBase _rb;
        private Rigidbody _body;
        private readonly List<Transform> _fuel = new List<Transform>();
        private StreamWriter _out;
        private string _path;
        private int _steps, _records;
        private float _lastTimer = -1f;
        private float _endSince = -1f;
        private int _match;

        private static string F(float f) => f.ToString("0.###", Inv);

        private void FixedUpdate()
        {
            var gm = BaseGameManager.Instance;
            if (gm == null) { CloseFile(); _robot = null; return; }

            if (_robot == null)
            {
                var rsc = FindFirstObjectByType<RobotSpawnController>();
                _robot = rsc?.BlueSpawnedRobots?.FirstOrDefault(r => r != null);
                if (_robot == null) return;
                _rb = _robot.GetComponent<RebuiltRobotBase>();
                _body = _robot.GetComponent<Rigidbody>();
                RefreshFuel();
            }

            // new match: timer jumped back up (ResetMatch / fresh scene)
            float timer = gm.Timer;
            if ((_out == null && _match == 0) || timer > _lastTimer + 1f)
            {
                CloseFile();
                OpenFile();
                RefreshFuel();
            }
            _lastTimer = timer;

            // stop at the end of the match: keep ~5 s after the buzzer (balls in flight still score), then close,
            // instead of writing t = 0 rows until the next match starts
            if (gm.GameState == MoSimCore.Enums.GameState.End)
            {
                if (_endSince < 0f) _endSince = Time.time;
                if (Time.time - _endSince > 5f) { CloseFile(); return; }
            }
            else _endSince = -1f;
            if (_out == null) return;

            if (++_steps < StepsPerRecord) return;
            _steps = 0;
            try { Write(gm); } catch (Exception e) { Log.Write("recorder write failed: " + e.Message); }
        }

        private void RefreshFuel()
        {
            _fuel.Clear();
            foreach (var c in FindObjectsByType<RebuiltGamePieceController>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
                _fuel.Add(c.transform);
        }

        private void OpenFile()
        {
            _match++;
            string dir = Path.Combine(Entry.RunDir, "demos");
            Directory.CreateDirectory(dir);
            _path = Path.Combine(dir, $"demo-{DateTime.Now:yyyyMMdd-HHmmss}-m{_match}.jsonl");
            _out = new StreamWriter(_path, false, new UTF8Encoding(false));
            _records = 0;
            Log.Write($"recording match {_match} -> {_path} (team {_rb?.TeamNumber})");
        }

        private void CloseFile()
        {
            if (_out == null) return;
            try { _out.Flush(); _out.Dispose(); } catch { }
            Log.Write($"recording closed: {_path} ({_records} records)");
            _out = null;
        }

        private void OnDestroy() => CloseFile();
        private void OnApplicationQuit() => CloseFile();

        private void Write(BaseGameManager gm)
        {
            if (_robot == null) return;
            var tr = _robot.transform;
            Vector3 p = tr.position, e = tr.eulerAngles;
            Vector3 v = _body != null ? _body.velocity : Vector3.zero;
            float wy = _body != null ? _body.angularVelocity.y : 0f;

            var sb = new StringBuilder(1024);
            sb.Append("{\"t\":").Append(F(gm.Timer));
            sb.Append(",\"gs\":").Append((int)gm.GameState).Append(",\"rs\":").Append((int)gm.RobotState);
            sb.Append(",\"blue\":").Append(RebuiltScoreUI.TotalBlueScore);
            sb.Append(",\"blueAuto\":").Append(RebuiltScoreUI.blueAutoFuelScore);
            sb.Append(",\"hub\":").Append((int)RebuiltShifts.ActiveHub);
            sb.Append(",\"wonAuto\":").Append((int)RebuiltShifts.WonAuto);
            sb.Append(",\"x\":").Append(F(p.x)).Append(",\"z\":").Append(F(p.z)).Append(",\"y\":").Append(F(p.y));
            sb.Append(",\"yaw\":").Append(F(e.y)).Append(",\"pitch\":").Append(F(e.x)).Append(",\"roll\":").Append(F(e.z));
            sb.Append(",\"vx\":").Append(F(v.x)).Append(",\"vz\":").Append(F(v.z)).Append(",\"wy\":").Append(F(wy));

            int held = 0;
            var fuelSb = (_records % FuelEvery == 0) ? new StringBuilder(12000) : null;
            bool first = true;
            foreach (var f in _fuel)
            {
                if (f == null) continue;
                Vector3 fp = f.position;
                Vector3 lp = tr.InverseTransformPoint(fp);
                if (Mathf.Abs(lp.x) < 0.45f && lp.z > -0.36f && lp.z < 0.62f && lp.y > 0.11f && lp.y < 0.60f) { held++; continue; }
                if (fuelSb == null) continue;
                if (!first) fuelSb.Append(',');
                first = false;
                fuelSb.Append('[').Append(F(fp.x)).Append(',').Append(F(fp.z)).Append(',').Append(F(fp.y)).Append(']');
            }
            sb.Append(",\"held\":").Append(held);
            try
            {
                var rt = _rb != null ? _rb.GetType() : null;
                const System.Reflection.BindingFlags BF = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
                var fDep = rt?.GetField("intakeDeployed", BF); var fJoint = rt?.GetField("intakeJoint", BF); var fAxis = rt?.GetField("intakeAxis", BF);
                if (fDep != null)
                {
                    sb.Append(",\"mech\":{\"dep\":").Append((bool)fDep.GetValue(_rb) ? 1 : 0);
                    var joint = fJoint?.GetValue(_rb); var axis = fAxis?.GetValue(_rb);
                    var m = joint?.GetType().GetMethod("GetAxisLocation");
                    if (m != null && axis != null) sb.Append(",\"slide\":").Append(F((float)m.Invoke(joint, new[] { axis })));
                    sb.Append('}');
                }
            }
            catch (Exception) { }

            // driver inputs (raw; camera yaw lets us convert TranslateAction to the field frame later)
            if (_rb != null)
            {
                Vector2 tv = _rb.TranslateAction != null ? _rb.TranslateAction.ReadValue<Vector2>() : Vector2.zero;
                float rv = _rb.RotateAction != null ? _rb.RotateAction.ReadValue<float>() : 0f;
                float camYaw = _rb.ThirdPersonCam?.CameraObject != null ? _rb.ThirdPersonCam.CameraObject.transform.eulerAngles.y : float.NaN;
                sb.Append(",\"in\":{\"tx\":").Append(F(tv.x)).Append(",\"ty\":").Append(F(tv.y)).Append(",\"rot\":").Append(F(rv));
                sb.Append(",\"cam\":").Append(float.IsNaN(camYaw) ? "null" : F(camYaw));
                sb.Append(",\"fc\":").Append(_rb.IsFieldCentric ? 1 : 0);
                sb.Append(",\"b\":[").Append(P(_rb.IntakeAction)).Append(',').Append(P(_rb.OuttakeAction)).Append(',')
                  .Append(P(_rb.AutoPass)).Append(',').Append(P(_rb.ManualShoot)).Append(',').Append(P(_rb.RobotSpecial)).Append("]}");
            }
            if (fuelSb != null) sb.Append(",\"fuel\":[").Append(fuelSb).Append(']');
            sb.Append('}');
            _out.WriteLine(sb.ToString());
            if (++_records % 50 == 0) _out.Flush();
        }

        private static int P(MoSimLib.MoSimInput i)
        {
            try { return i != null && i.IsPressed() ? 1 : 0; } catch { return 0; }
        }
    }
}
