// Lockstep TCP bridge: Python drives 4414 through this component.
//
// Protocol: newline-delimited JSON over 127.0.0.1:<port> (default 47414, override with -mosimrl-port N).
//   {"cmd":"hello"}                          -> {"ok":true, info...}
//   {"cmd":"config","timeScale":2,"decisionSteps":22,"cameras":0,"fps":-1}
//                                            -> {"ok":true, info...}
//   {"cmd":"reset"}                          -> first state of a fresh match (loads the game scene if needed)
//   {"cmd":"act","v":[vx,vz],"rot":r,"b":[intake,shoot,pass,manual,special]}
//                                            -> state after `decisionSteps` physics steps
//   {"cmd":"release"}                        -> {"ok":true}; human controls return
//
// Why the physics thread BLOCKS on the socket at each decision point: without it Python's view and the
// simulation drift apart and (state, action, next state) stop being a well-defined transition. Game time
// is still driven by Unity's own clock, so a slow client only lengthens wall time, not the match.

using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using Games.Rebuilt.FieldScripts;
using Games.Rebuilt.GamePieceSystem;
using Games.Rebuilt.Robots;
using Games.Rebuilt.Scoring;
using GameSystems.Management;
using MoSimCore.BaseClasses.GameManagement;
using MoSimCore.Enums;
using MoSimCore.GameSelection;
using RobotFramework.Controllers.Drivetrain;
using UnityEngine;
using MoSimSceneManager = MoSimCore.SceneTransitions.SceneManager;

namespace MoSimRL
{
    [Serializable]
    public class BridgeMsg
    {
        public string cmd = "";
        public float[] v;
        public float rot;
        public int[] b;
        public float timeScale = -1f;
        public int decisionSteps = -1;
        public int cameras = -1;
        public int fps = -2;
        public int rff = 0;            // 1: also feed the rotate command to the robot's RotateAction (turret feed-forward)
        public int stepsPerFrame = -1; // >0: fixed frame step of N physics steps (Time.captureDeltaTime), CPU-bound; 0: real-time pacing
        public int lite = -1;          // act: 1 = omit the per-ball arrays from replies (planner rollouts), 0 = full state
        public string blob;            // restore: base64 snapshot from another instance
    }

    public class Bridge : MonoBehaviour
    {
        public static Bridge Instance { get; private set; }

        // the driver inputs in force at a snapshot, so a restored instance keeps pressing what the source pressed
        public struct PadState
        {
            public bool[] Pressed; public Vector2 V; public float Rot; public bool Rff;
            public void Write(System.IO.BinaryWriter w)
            {
                for (int i = 0; i < 5; i++) w.Write(Pressed != null && i < Pressed.Length && Pressed[i]);
                w.Write(V.x); w.Write(V.y); w.Write(Rot); w.Write(Rff);
            }
            public static PadState Read(System.IO.BinaryReader r)
            {
                var p = new PadState { Pressed = new bool[5] };
                for (int i = 0; i < 5; i++) p.Pressed[i] = r.ReadBoolean();
                p.V = new Vector2(r.ReadSingle(), r.ReadSingle()); p.Rot = r.ReadSingle(); p.Rff = r.ReadBoolean();
                return p;
            }
        }
        private bool _lite;
        private bool _rff;

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        // carried-ball envelope in robot-local metres: hopper (hightide_contract.md §6) plus the deployed
        // intake slide, where calibration found balls riding at z 0.35..0.54, y 0.23..0.36
        private const float HopperX = 0.45f, HopperZMin = -0.36f, HopperZMax = 0.62f, HopperYMin = 0.11f, HopperYMax = 0.60f;

        private TcpListener _listener;
        private TcpClient _client;
        private float _lastMute = -10f;
        private NetworkStream _stream;
        private readonly List<byte> _rx = new List<byte>();
        private readonly byte[] _buf = new byte[65536];

        private int _decisionSteps = 22;           // 22 x 4.5 ms ≈ 0.1 s of game time
        private bool _camerasOff;

        private bool _controlling;
        private bool _awaitingAction;
        private int _stepCounter;
        private bool _resetting;
        private float _endSince = -1f;

        private GameObject _robot;
        private Rigidbody _rb;
        private DriveController _drive;
        private RebuiltRobotBase _robotBase;
        private ButtonInjector _buttons;
        private readonly bool[] _pressed = new bool[5];
        private Vector2 _cmdV;
        private float _cmdRot;
        private readonly List<Transform> _fuel = new List<Transform>();
        private int _episode;

        public static int Port() => IntArg("-mosimrl-port", 47414);

        private static int IntArg(string name, int dflt)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == name && int.TryParse(args[i + 1], out int p)) return p;
            return dflt;
        }

        // Orphan guard: bot instances are started via `open`, so they outlive the Python process that launched them,
        // and a headless player ignores the macOS quit event — orphans once blocked a system restart. With
        // -mosimrl-owner <pid> the instance quits by itself within ~2 s of that process disappearing.
        [DllImport("/usr/lib/libSystem.B.dylib", SetLastError = true)]
        private static extern int kill(int pid, int sig);
        private const int ESRCH = 3;
        private int _ownerPid = -1;
        private float _lastOwnerCheck;
        private float _quitAt = -1f;

        private void CheckOwner()
        {
            if (_quitAt >= 0f)
            {
                // Application.Quit is only a request; if the player is still here, end it hard
                if (Time.realtimeSinceStartup - _quitAt > 10f) kill(System.Diagnostics.Process.GetCurrentProcess().Id, 9);
                return;
            }
            if (_ownerPid <= 0 || Time.realtimeSinceStartup - _lastOwnerCheck < 2f) return;
            _lastOwnerCheck = Time.realtimeSinceStartup;
            if (kill(_ownerPid, 0) == 0 || Marshal.GetLastWin32Error() != ESRCH) return;
            Log.Write($"owner process {_ownerPid} is gone; quitting");
            _quitAt = Time.realtimeSinceStartup;
            ReleaseControl();
            Application.Quit();
        }

        private void Awake()
        {
            Instance = this;
            AudioListener.volume = 0f;
            AudioListener.pause = true;
            int port = Port();
            _ownerPid = IntArg("-mosimrl-owner", -1);
            if (_ownerPid > 0) Log.Write($"orphan guard: quits when process {_ownerPid} exits");
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, port);
                _listener.Start();
                Log.Write($"bridge listening on 127.0.0.1:{port}");
            }
            catch (Exception e)
            {
                Log.Write($"bridge listen failed on {port}: {e.Message}");
            }
        }

        private void OnDestroy()
        {
            ReleaseControl();
            try { _client?.Close(); } catch { }
            try { _listener?.Stop(); } catch { }
        }

        // ---------------------------------------------------------------- socket plumbing

        private void Send(string json)
        {
            if (_stream == null) return;
            var bytes = Encoding.UTF8.GetBytes(json + "\n");
            try { _stream.Write(bytes, 0, bytes.Length); }
            catch (Exception e) { Log.Write("send failed: " + e.Message); DropClient(); }
        }

        private void DropClient()
        {
            ReleaseControl();
            try { _client?.Close(); } catch { }
            _client = null; _stream = null; _rx.Clear();
        }

        private string TryPopLine()
        {
            int nl = _rx.IndexOf((byte)'\n');
            if (nl < 0) return null;
            string line = Encoding.UTF8.GetString(_rx.GetRange(0, nl).ToArray());
            _rx.RemoveRange(0, nl + 1);
            return line;
        }

        private string ReadLineNonBlocking()
        {
            if (_stream == null) return null;
            try
            {
                while (_client.Available > 0)
                {
                    int n = _stream.Read(_buf, 0, Math.Min(_buf.Length, _client.Available));
                    if (n <= 0) break;
                    for (int i = 0; i < n; i++) _rx.Add(_buf[i]);
                }
            }
            catch (Exception e) { Log.Write("read failed: " + e.Message); DropClient(); return null; }
            return TryPopLine();
        }

        private string ReadLineBlocking(int timeoutMs)
        {
            string line = TryPopLine();
            if (line != null || _stream == null) return line;
            _stream.ReadTimeout = timeoutMs;
            try
            {
                while (true)
                {
                    int n = _stream.Read(_buf, 0, _buf.Length);
                    if (n <= 0) { DropClient(); return null; }
                    for (int i = 0; i < n; i++) _rx.Add(_buf[i]);
                    line = TryPopLine();
                    if (line != null) return line;
                }
            }
            catch (Exception e) { Log.Write("blocking read ended: " + e.Message); DropClient(); return null; }
        }

        // ---------------------------------------------------------------- main-thread command handling

        private void Update()
        {
            CheckOwner();
            // Single-client bridge, newest connection wins: a peer that went away without us noticing
            // (script exited) must not block the next one.
            if (_listener != null && _listener.Pending())
            {
                if (_client != null) { Log.Write("new client connecting; dropping previous connection"); DropClient(); }
                _client = _listener.AcceptTcpClient();
                _client.NoDelay = true;
                _stream = _client.GetStream();
                Log.Write("bridge client connected");
            }

            if (_camerasOff) foreach (var cam in Camera.allCameras) cam.enabled = false;
            // bot sessions are silent: 8 headless instances playing match audio is just noise for the human at the desk
            AudioListener.volume = 0f;
            AudioListener.pause = true;
            if (Time.unscaledTime - _lastMute > 1f)
            {
                _lastMute = Time.unscaledTime;
                foreach (var src in FindObjectsByType<AudioSource>(FindObjectsInactive.Include, FindObjectsSortMode.None)) src.mute = true;
            }

            if (_controlling || _resetting) return;
            string line = ReadLineNonBlocking();
            if (line != null) Handle(line);
        }

        private void Handle(string line)
        {
            BridgeMsg m;
            try { m = JsonUtility.FromJson<BridgeMsg>(line); }
            catch (Exception e) { Send(Err("bad json: " + e.Message)); return; }

            switch (m.cmd)
            {
                case "hello": Send(Info()); break;
                case "config": ApplyConfig(m); Send(Info()); break;
                case "reset": StartCoroutine(ResetEpisode()); break;
                case "release": ReleaseControl(); Send("{\"ok\":true}"); break;
                case "act": Send(Err("no episode running; send reset first")); break;
                default: Send(Err("unknown cmd " + m.cmd)); break;
            }
        }

        private void ApplyConfig(BridgeMsg m)
        {
            if (m.timeScale > 0f) Time.timeScale = m.timeScale;
            if (m.decisionSteps > 0) _decisionSteps = m.decisionSteps;
            if (m.cameras == 0) _camerasOff = true;
            if (m.cameras == 1) { _camerasOff = false; foreach (var cam in Resources.FindObjectsOfTypeAll<Camera>()) cam.enabled = true; }
            if (m.fps > -2) { QualitySettings.vSyncCount = 0; Application.targetFrameRate = m.fps; }
            // Fixed frame step: every frame advances exactly N fixed steps of game time no matter how long it took, so
            // the Update/FixedUpdate interleave (Update-sampled scoring, the match timer) is the same on every run and
            // the game runs as fast as the CPU allows instead of at timeScale x real time.
            if (m.stepsPerFrame > 0) { Time.captureDeltaTime = Time.fixedDeltaTime * m.stepsPerFrame; QualitySettings.vSyncCount = 0; Application.targetFrameRate = -1; }
            if (m.stepsPerFrame == 0) Time.captureDeltaTime = 0f;
            Log.Write($"config timeScale={Time.timeScale} decisionSteps={_decisionSteps} camerasOff={_camerasOff} fps={Application.targetFrameRate} captureDeltaTime={Time.captureDeltaTime}");
        }

        private string Info()
        {
            return "{\"ok\":true,\"version\":\"" + Application.version + "\"" +
                   ",\"fixedDeltaTime\":" + F(Time.fixedDeltaTime) +
                   ",\"timeScale\":" + F(Time.timeScale) +
                   ",\"captureDeltaTime\":" + Time.captureDeltaTime.ToString("0.#######", Inv) +
                   ",\"decisionSteps\":" + _decisionSteps +
                   ",\"camerasOff\":" + (_camerasOff ? "true" : "false") +
                   ",\"inMatch\":" + (BaseGameManager.Instance != null ? "true" : "false") + "}";
        }

        private static string Err(string msg) => "{\"ok\":false,\"error\":\"" + msg.Replace("\"", "'") + "\"}";

        // ---------------------------------------------------------------- episode lifecycle

        private IEnumerator ResetEpisode()
        {
            _resetting = true;
            ReleaseControl();

            if (BaseGameManager.Instance == null)
            {
                float t0 = Time.realtimeSinceStartup;
                while ((SelectedGameManager.Instance?.SelectedGame == null || MoSimSceneManager.Instance == null) &&
                       Time.realtimeSinceStartup - t0 < 60f)
                    yield return null;
                var game = SelectedGameManager.Instance?.SelectedGame;
                if (game == null) { _resetting = false; Send(Err("no selected game")); yield break; }
                Log.Write($"reset: loading scene '{game.GameScene?.Name}'");
                MoSimSceneManager.Instance.LoadScene(game.GameScene, "CrossFade");
                t0 = Time.realtimeSinceStartup;
                while ((BaseGameManager.Instance == null || FindBlueRobot() == null) && Time.realtimeSinceStartup - t0 < 60f)
                    yield return null;
                if (BaseGameManager.Instance == null) { _resetting = false; Send(Err("match scene never loaded")); yield break; }
                yield return new WaitForSecondsRealtime(1f);
            }

            var gm = BaseGameManager.Instance;
            while (gm.IsResetting) yield return null;
            gm.StartCoroutine(gm.ResetMatch());
            yield return null;
            while (gm.IsResetting) yield return null;

            // wait for the fresh robot and the match clock to start in auto with the robot enabled
            float w0 = Time.realtimeSinceStartup;
            while (Time.realtimeSinceStartup - w0 < 30f &&
                   (FindBlueRobot() == null || gm.GameState != GameState.Auto || gm.RobotState != RobotState.Enabled))
                yield return null;

            _robot = FindBlueRobot();
            if (_robot == null) { _resetting = false; Send(Err("no blue robot after reset")); yield break; }
            _rb = _robot.GetComponent<Rigidbody>();
            _drive = _robot.GetComponent<DriveController>();
            _robotBase = _robot.GetComponent<RebuiltRobotBase>();
            try { _buttons = ButtonInjector.Install(_robotBase); }
            catch (Exception e) { Log.Write("button install failed: " + e); _buttons = null; }

            Snapshot.BuildKeys();
            _fuel.Clear();
            // fuel ids in hierarchy-key order, so `fid` means the same ball in every instance
            foreach (var c in FindObjectsByType<RebuiltGamePieceController>(FindObjectsInactive.Exclude, FindObjectsSortMode.None)
                         .OrderBy(c => Snapshot.KeyOf(c.gameObject) ?? "~", StringComparer.Ordinal))
                _fuel.Add(c.transform);

            _cmdV = Vector2.zero; _cmdRot = 0f;
            for (int i = 0; i < _pressed.Length; i++) _pressed[i] = false;
            _endSince = -1f;
            _stepCounter = 0;
            _episode++;
            _controlling = true;
            _resetting = false;
            Log.Write($"snapshot keys={Snapshot.KeyCount} duplicates={Snapshot.DuplicateKeys}");
            Log.Write($"episode {_episode} start: robot={_robot.name} team={_robotBase?.TeamNumber} fuel={_fuel.Count} " +
                      $"pos=({_robot.transform.position.x:F3},{_robot.transform.position.z:F3}) buttons={(_buttons != null)}");
            Send(State(false));
            _awaitingAction = true;
        }

        private void ReleaseControl()
        {
            _controlling = false;
            _awaitingAction = false;
            if (_buttons != null) { try { _buttons.Dispose(); } catch { } _buttons = null; }
        }

        private static GameObject FindBlueRobot()
        {
            var rsc = FindFirstObjectByType<RobotSpawnController>();
            return rsc?.BlueSpawnedRobots?.FirstOrDefault(r => r != null);
        }

        // ---------------------------------------------------------------- physics-step control loop

        private void FixedUpdate()
        {
            if (!_controlling) return;
            if (_robot == null) { Log.Write("robot vanished mid-episode"); ReleaseControl(); Send(Err("robot vanished")); return; }

            while (_awaitingAction)
            {
                string line = ReadLineBlocking(300000);
                if (line == null) { Log.Write("no action within 300 s; releasing control"); ReleaseControl(); return; }
                BridgeMsg m;
                try { m = JsonUtility.FromJson<BridgeMsg>(line); } catch { m = null; }
                // snapshot / restore happen at the decision point and keep the instance waiting for its next action
                if (m != null && m.cmd == "snapshot") { SendSnapshot(); continue; }
                if (m != null && m.cmd == "restore") { DoRestore(m.blob); continue; }
                if (m != null && m.cmd == "keys") { SendKeys(); continue; }
                if (m == null || m.cmd != "act")
                {
                    // hand anything else (reset/release/config/hello) to the main-thread handler
                    ReleaseControl();
                    if (m != null) Handle(line);
                    return;
                }
                ApplyAction(m);
                _awaitingAction = false;
                _stepCounter = 0;
            }

            _drive?.overideInput(_cmdV, _cmdRot, DriveController.DriveMode.FieldOriented);

            var gm = BaseGameManager.Instance;
            if (gm != null && gm.GameState == GameState.End && _endSince < 0f) _endSince = Time.time;
            // HubScoring keeps counting for 3 s after the buzzer; include that tail in the episode.
            bool done = _endSince >= 0f && Time.time - _endSince >= 3.5f;

            if (++_stepCounter >= _decisionSteps || done)
            {
                Send(State(done));
                if (done) { ReleaseControl(); return; }
                _awaitingAction = true;
            }
        }

        private void ApplyAction(BridgeMsg m)
        {
            if (m.lite >= 0) _lite = m.lite == 1;
            _rff = m.rff == 1;
            _cmdV = (m.v != null && m.v.Length >= 2) ? new Vector2(Mathf.Clamp(m.v[0], -1f, 1f), Mathf.Clamp(m.v[1], -1f, 1f)) : Vector2.zero;
            _cmdRot = Mathf.Clamp(m.rot, -1f, 1f);
            if (_buttons == null || m.b == null) return;
            for (int i = 0; i < _pressed.Length && i < m.b.Length; i++) _pressed[i] = m.b[i] != 0;
            // Re-send the whole pad state every decision, not only on change: the first press of an episode was
            // observed to get lost (intake never engaged until the demo toggled it again), and a full-state
            // event every 0.1 s self-heals any dropped event.
            _buttons.SetAll(_pressed, m.rff == 1 ? _cmdRot : 0f);
        }

        // ---------------------------------------------------------------- snapshot / restore (lookahead planning)

        private void SendSnapshot()
        {
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var pad = new PadState { Pressed = (bool[])_pressed.Clone(), V = _cmdV, Rot = _cmdRot, Rff = _rff };
                byte[] blob = Snapshot.Capture(pad, out string summary);
                Send("{\"ok\":true,\"t\":" + F(BaseGameManager.Instance != null ? BaseGameManager.Instance.Timer : -1f) +
                     ",\"ms\":" + sw.ElapsedMilliseconds + ",\"summary\":\"" + summary + "\",\"blob\":\"" + Convert.ToBase64String(blob) + "\"}");
            }
            catch (Exception e) { Log.Write("snapshot failed: " + e); Send(Err("snapshot failed: " + e.Message)); }
        }

        private void DoRestore(string b64)
        {
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                string res = Snapshot.Restore(Convert.FromBase64String(b64), out PadState pad);
                for (int i = 0; i < _pressed.Length; i++) _pressed[i] = pad.Pressed[i];
                _cmdV = pad.V; _cmdRot = pad.Rot; _rff = pad.Rff;
                if (_buttons != null)
                {
                    _buttons.SetAll(_pressed, _rff ? _cmdRot : 0f);
                    try { UnityEngine.InputSystem.InputSystem.Update(); } catch (Exception e) { Log.Write("input update after restore: " + e.Message); }
                }
                _endSince = -1f;
                _stepCounter = 0;
                _restoreInfo = res + $" ms={sw.ElapsedMilliseconds}";
                Send(State(false));
            }
            catch (Exception e) { Log.Write("restore failed: " + e); Send(Err("restore failed: " + e.Message)); }
        }

        private void SendKeys()
        {
            var sb = new StringBuilder(65536);
            sb.Append("{\"ok\":true,\"keys\":").Append(Snapshot.KeyCount).Append(",\"dups\":").Append(Snapshot.DuplicateKeys).Append(",\"balls\":[");
            bool first = true;
            foreach (var b in FindObjectsByType<RebuiltGamePieceController>(FindObjectsInactive.Include, FindObjectsSortMode.None)
                         .Select(b => (k: Snapshot.KeyOf(b.gameObject), b)).OrderBy(p => p.k, StringComparer.Ordinal))
            {
                if (!first) sb.Append(',');
                first = false;
                var fp = b.b.transform.position;
                sb.Append("[\"").Append(b.k ?? "?").Append("\",").Append(F(fp.x)).Append(',').Append(F(fp.z)).Append(',').Append(F(fp.y)).Append(']');
            }
            sb.Append("]}");
            Send(sb.ToString());
        }

        private string _restoreInfo;

        // ---------------------------------------------------------------- state serialisation

        private static string F(float f) => f.ToString("0.###", Inv);

        private string State(bool done)
        {
            var gm = BaseGameManager.Instance;
            var sb = new StringBuilder(16384);
            sb.Append("{\"ok\":true,\"ep\":").Append(_episode);
            sb.Append(",\"done\":").Append(done ? "true" : "false");
            sb.Append(",\"t\":").Append(F(gm != null ? gm.Timer : -1f));
            sb.Append(",\"gs\":").Append(gm != null ? (int)gm.GameState : -1);
            sb.Append(",\"rs\":").Append(gm != null ? (int)gm.RobotState : -1);
            sb.Append(",\"blue\":").Append(RebuiltScoreUI.TotalBlueScore);
            sb.Append(",\"red\":").Append(RebuiltScoreUI.TotalRedScore);
            sb.Append(",\"blueAuto\":").Append(RebuiltScoreUI.blueAutoFuelScore);
            sb.Append(",\"hub\":").Append((int)RebuiltShifts.ActiveHub);
            sb.Append(",\"shift\":").Append(RebuiltShifts.ShiftNumber);
            sb.Append(",\"toShift\":").Append(F(RebuiltShifts.SecondsUntilNextShift));
            sb.Append(",\"wonAuto\":").Append((int)RebuiltShifts.WonAuto);
            sb.Append(",\"gameTime\":").Append(F(Time.time));
            sb.Append(",\"frame\":").Append(Time.frameCount);      // frames per decision = how finely Update-sampled scoring sees the game
            if (_restoreInfo != null) { sb.Append(",\"restored\":\"").Append(_restoreInfo).Append('"'); _restoreInfo = null; }

            if (_robot != null)
            {
                var tr = _robot.transform;
                Vector3 p = tr.position, e = tr.eulerAngles;
                Vector3 v = _rb != null ? _rb.velocity : Vector3.zero;
                Vector3 w = _rb != null ? _rb.angularVelocity : Vector3.zero;
                sb.Append(",\"robot\":{\"x\":").Append(F(p.x)).Append(",\"y\":").Append(F(p.y)).Append(",\"z\":").Append(F(p.z));
                sb.Append(",\"yaw\":").Append(F(e.y)).Append(",\"pitch\":").Append(F(e.x)).Append(",\"roll\":").Append(F(e.z));
                sb.Append(",\"vx\":").Append(F(v.x)).Append(",\"vz\":").Append(F(v.z)).Append(",\"wy\":").Append(F(w.y));
                sb.Append(",\"inZone\":").Append(_robotBase != null && SafeInZone() ? "true" : "false");
                sb.Append('}');
                // intake mechanism (Hightide private state via reflection): latched deploy flag, slide position, kicker lock
                try
                {
                    var rt = _robotBase != null ? _robotBase.GetType() : null;
                    const BindingFlags BF = BindingFlags.Instance | BindingFlags.NonPublic;
                    var fDep = rt?.GetField("intakeDeployed", BF); var fKick = rt?.GetField("kickerLocked", BF);
                    var fJoint = rt?.GetField("intakeJoint", BF); var fAxis = rt?.GetField("intakeAxis", BF);
                    if (fDep != null)
                    {
                        sb.Append(",\"mech\":{\"dep\":").Append((bool)fDep.GetValue(_robotBase) ? 1 : 0);
                        if (fKick != null) sb.Append(",\"kick\":").Append((bool)fKick.GetValue(_robotBase) ? 1 : 0);
                        var joint = fJoint?.GetValue(_robotBase);
                        var axis = fAxis?.GetValue(_robotBase);
                        var m = joint?.GetType().GetMethod("GetAxisLocation");
                        if (m != null && axis != null) sb.Append(",\"slide\":").Append(F((float)m.Invoke(joint, new[] { axis })));
                        var kj = rt.GetField("kickerBarJoint", BF)?.GetValue(_robotBase);
                        var ga = kj?.GetType().GetMethod("GetAngle", Type.EmptyTypes);
                        if (ga != null) sb.Append(",\"kang\":").Append(F((float)ga.Invoke(kj, null)));
                        sb.Append('}');
                    }
                    // AutoShoot feed gates (Hightide.Shooter): turret error / unwinding, bump latch, trench, bump strip
                    var turret = rt?.GetField("turret", BF)?.GetValue(_robotBase);
                    if (turret != null)
                    {
                        var tt = turret.GetType();
                        float err = (float)tt.GetMethod("GetError").Invoke(turret, null);
                        bool wrap = (bool)tt.GetMethod("IsWrapping").Invoke(turret, null);
                        bool bump = (bool)rt.GetField("bumpCrossingLatched", BF).GetValue(_robotBase);
                        bool trench = (bool)rt.GetMethod("TurretUnderTrench", BF).Invoke(_robotBase, null);
                        bool bz = (bool)rt.GetMethod("InBumpZone").Invoke(_robotBase, null);
                        sb.Append(",\"gate\":{\"err\":").Append(F(err)).Append(",\"wrap\":").Append(wrap ? 1 : 0)
                          .Append(",\"bump\":").Append(bump ? 1 : 0).Append(",\"trench\":").Append(trench ? 1 : 0)
                          .Append(",\"bz\":").Append(bz ? 1 : 0).Append('}');
                    }
                }
                catch (Exception) { }
                // what the robot's own MoSimInputs report right now (after our injection) — to verify presses land
                if (_buttons != null)
                {
                    sb.Append(",\"bIn\":[");
                    for (int i = 0; i < 5; i++)
                    {
                        if (i > 0) sb.Append(',');
                        bool on; try { on = _buttons.ReadBack((Button)i); } catch { on = false; }
                        sb.Append(on ? 1 : 0);
                    }
                    sb.Append(']');
                    try { sb.Append(",\"rotIn\":").Append(F(_buttons.RotateReadBack())); } catch { }
                }

                if (_lite)
                {
                    // planner rollouts: held count only, no per-ball arrays
                    int h = 0;
                    for (int i = 0; i < _fuel.Count; i++)
                    {
                        var f = _fuel[i];
                        if (f == null) continue;
                        Vector3 lp = tr.InverseTransformPoint(f.position);
                        if (Mathf.Abs(lp.x) < HopperX && lp.z > HopperZMin && lp.z < HopperZMax && lp.y > HopperYMin && lp.y < HopperYMax) h++;
                    }
                    sb.Append(",\"held\":").Append(h).Append('}');
                    return sb.ToString();
                }
                int held = 0;
                var fid = new StringBuilder(2048);          // stable per-ball id (index into _fuel) for each fuel entry
                sb.Append(",\"fuel\":[");
                bool first = true;
                for (int i = 0; i < _fuel.Count; i++)
                {
                    var f = _fuel[i];
                    if (f == null) continue;
                    Vector3 fp = f.position;
                    Vector3 lp = tr.InverseTransformPoint(fp);
                    bool inHopper = Mathf.Abs(lp.x) < HopperX && lp.z > HopperZMin && lp.z < HopperZMax && lp.y > HopperYMin && lp.y < HopperYMax;
                    if (inHopper) { held++; continue; }
                    if (!first) { sb.Append(','); fid.Append(','); }
                    first = false;
                    sb.Append('[').Append(F(fp.x)).Append(',').Append(F(fp.z)).Append(',').Append(F(fp.y)).Append(']');
                    fid.Append(i);
                }
                sb.Append("],\"held\":").Append(held);
                sb.Append(",\"fid\":[").Append(fid).Append(']');
            }
            sb.Append('}');
            return sb.ToString();
        }

        private bool SafeInZone()
        {
            try { return _robotBase.InAllianceZone(); } catch { return false; }
        }
    }
}
