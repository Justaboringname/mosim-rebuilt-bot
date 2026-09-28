// MoSimRL P0 probe — answers: can our code run while the BUILT-IN 4414 is the selected robot?
// Also measures: shipped fixedDeltaTime, HubScoring instance count, ResetMatch drift, timeScale throughput,
// RollerThunker exception rate, and whether DriveController.overideInput moves the robot.
//
// Loaded by a bootstrap in a host mod DLL. Inert unless run/ENABLE exists.
// run/PROBE runs this one-shot probe in the first match; run/BRIDGE starts the Python control bridge (Bridge.cs).

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Games.Rebuilt.FieldScripts;
using Games.Rebuilt.Scoring;
using GameSystems.Management;
using MoSimCore.BaseClasses.GameManagement;
using MoSimCore.GameSelection;
using RobotFramework;
using RobotFramework.Controllers.Drivetrain;
using UnityEngine;
using UnitySceneManager = UnityEngine.SceneManagement.SceneManager;
using MoSimSceneManager = MoSimCore.SceneTransitions.SceneManager;
using Scene = UnityEngine.SceneManagement.Scene;
using LoadSceneMode = UnityEngine.SceneManagement.LoadSceneMode;

namespace MoSimRL
{
    public static class Entry
    {
        public static readonly string RunDir = ResolveRunDir();

        // Where the flag files (ENABLE / BRIDGE / RECORD / ...) live: the `-mosimrl-run <dir>` launch argument (bot
        // instances; gamectl passes it), else the path tools/install-hook.sh wrote to ~/.mosimrl_run (a human session
        // started from Steam), else ~/Projects/mosim-rl/run. The harness is loaded from bytes, so it cannot use its
        // own file location.
        private static string ResolveRunDir()
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            try
            {
                var args = Environment.GetCommandLineArgs();
                for (int i = 0; i + 1 < args.Length; i++)
                    if (args[i] == "-mosimrl-run") return args[i + 1];
                string ptr = Path.Combine(home, ".mosimrl_run");
                if (File.Exists(ptr)) { string p = File.ReadAllText(ptr).Trim(); if (p.Length > 0) return p; }
            }
            catch (Exception) { }
            return Path.Combine(home, "Projects", "mosim-rl", "run");
        }

        private static bool _inited;

        public static bool Flag(string name) => File.Exists(Path.Combine(RunDir, name));

        // Called from a MonoBehaviour constructor during AssetBundle deserialization:
        // pure managed work only — no Unity object APIs here.
        public static void Init()
        {
            if (_inited) return;
            _inited = true;
            bool enabled = Flag("ENABLE");
            Log.Write($"Init from host ctor (thread={System.Threading.Thread.CurrentThread.ManagedThreadId}, enabled={enabled})");
            if (!enabled) return;
            UnitySceneManager.sceneLoaded += OnSceneLoaded;
        }

        private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            try
            {
                if (Host.Instance == null)
                {
                    var go = new GameObject("MoSimRL.Host");
                    UnityEngine.Object.DontDestroyOnLoad(go);
                    go.AddComponent<Host>();
                }
                Host.Instance.OnScene(scene, mode);
            }
            catch (Exception e)
            {
                Log.Write("OnSceneLoaded failed: " + e);
            }
        }
    }

    public static class Log
    {
        private static readonly object Gate = new object();

        public static void Write(string msg)
        {
            string line = $"[MoSimRL] {DateTime.Now:HH:mm:ss.fff} {msg}";
            try { Debug.Log(line); } catch { }
            try
            {
                lock (Gate)
                {
                    Directory.CreateDirectory(Entry.RunDir);
                    File.AppendAllText(Path.Combine(Entry.RunDir, "probe.log"), line + "\n");
                }
            }
            catch { }
        }
    }

    public class Host : MonoBehaviour
    {
        public static Host Instance { get; private set; }

        private bool _autoStarted;
        private bool _matchProbeRunning;
        private readonly Dictionary<string, object> _summary = new Dictionary<string, object>();

        // exception accounting (logMessageReceivedThreaded can fire off-thread)
        private int _excTotal;
        private int _excRollerThunker;

        private void Awake()
        {
            Instance = this;
            Application.logMessageReceivedThreaded += OnLog;
            LogEnv("awake");
            _agentSession = Entry.Flag("BRIDGE") || Entry.Flag("PROBE");
            if (Entry.Flag("BRIDGE")) gameObject.AddComponent<Bridge>();
            if (Entry.Flag("RECORD")) gameObject.AddComponent<Recorder>();
        }

        private void OnDestroy()
        {
            Application.logMessageReceivedThreaded -= OnLog;
        }

        // No-upload guard for AGENT sessions (run/BRIDGE or run/PROBE): every match is labelled as cheated,
        // so SteamLeaderboards never submits (it checks wasCheated at GameState.End). StartMatch/ResetTimer
        // clear the flag, hence re-assert it every frame and every physics step. Human-only sessions
        // (run/RECORD alone) are left untouched so the driver's own scores count as usual.
        private bool _agentSession;

        private void AssertNoUpload()
        {
            if (!_agentSession) return;
            var gm = BaseGameManager.Instance;
            if (gm != null && !gm.wasCheated) gm.wasCheated = true;
        }

        private void Update() => AssertNoUpload();
        private void FixedUpdate() => AssertNoUpload();
        private void LateUpdate() => AssertNoUpload();

        private void OnLog(string condition, string stackTrace, LogType type)
        {
            if (type != LogType.Exception) return;
            System.Threading.Interlocked.Increment(ref _excTotal);
            if ((stackTrace != null && stackTrace.Contains("RollerThunker")) ||
                (condition != null && condition.Contains("RollerThunker")))
                System.Threading.Interlocked.Increment(ref _excRollerThunker);
        }

        private void LogEnv(string tag)
        {
            Log.Write($"env[{tag}] app={Application.version} unity={Application.unityVersion} " +
                      $"fixedDeltaTime={Time.fixedDeltaTime:F6} maximumDeltaTime={Time.maximumDeltaTime:F4} " +
                      $"timeScale={Time.timeScale} simulationMode={Physics.simulationMode} " +
                      $"solverIters={Physics.defaultSolverIterations}/{Physics.defaultSolverVelocityIterations} " +
                      $"targetFrameRate={Application.targetFrameRate} vSync={QualitySettings.vSyncCount} " +
                      $"batchmode={Application.isBatchMode} gfx={SystemInfo.graphicsDeviceType}");
            _summary["fixedDeltaTime_" + tag] = Time.fixedDeltaTime;
            _summary["maximumDeltaTime_" + tag] = Time.maximumDeltaTime;
            _summary["simulationMode_" + tag] = Physics.simulationMode.ToString();
        }

        public void OnScene(Scene scene, LoadSceneMode mode)
        {
            Log.Write($"sceneLoaded name='{scene.name}' index={scene.buildIndex} mode={mode} " +
                      $"gameManager={(BaseGameManager.Instance != null ? BaseGameManager.Instance.GetType().FullName : "null")}");

            if (Entry.Flag("AUTOSTART") && !_autoStarted && scene.name.IndexOf("Menu", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                _autoStarted = true;
                StartCoroutine(AutoStart());
            }

            if (Entry.Flag("PROBE") && !_matchProbeRunning && scene.name.IndexOf("Menu", StringComparison.OrdinalIgnoreCase) < 0)
            {
                _matchProbeRunning = true;
                StartCoroutine(MatchProbe());
            }
        }

        private IEnumerator AutoStart()
        {
            float deadline = Time.realtimeSinceStartup + 60f;
            while (Time.realtimeSinceStartup < deadline &&
                   (SelectedGameManager.Instance == null || SelectedGameManager.Instance.SelectedGame == null ||
                    MoSimSceneManager.Instance == null))
                yield return null;

            if (SelectedGameManager.Instance?.SelectedGame == null || MoSimSceneManager.Instance == null)
            {
                Log.Write("autostart: managers never became ready; giving up");
                yield break;
            }

            yield return new WaitForSecondsRealtime(3f);
            var game = SelectedGameManager.Instance.SelectedGame;
            Log.Write($"autostart: loading game '{game.name}' scene='{game.GameScene?.Name}'");
            MoSimSceneManager.Instance.LoadScene(game.GameScene, "CrossFade");
        }

        private static GameObject FindBlueRobot()
        {
            var rsc = FindFirstObjectByType<RobotSpawnController>();
            if (rsc == null || rsc.BlueSpawnedRobots == null) return null;
            return rsc.BlueSpawnedRobots.FirstOrDefault(r => r != null);
        }

        private AttachProbe Attach(GameObject robot)
        {
            var probe = robot.GetComponent<AttachProbe>();
            if (probe == null) probe = robot.AddComponent<AttachProbe>();
            var rb = robot.GetComponent<RobotBase>();
            Log.Write($"attach: robot='{robot.name}' id={robot.GetInstanceID()} type={rb?.GetType().FullName} " +
                      $"team={rb?.TeamNumber} alliance={rb?.Alliance} hasDrive={robot.GetComponent<DriveController>() != null}");
            return probe;
        }

        private string HubReport()
        {
            var hubs = FindObjectsByType<HubScoring>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            var f = typeof(HubScoring).GetField("occupyColliders", BindingFlags.Instance | BindingFlags.NonPublic);
            var sb = new StringBuilder($"hubs={hubs.Length}");
            foreach (var h in hubs)
            {
                var cols = f?.GetValue(h) as Collider[];
                sb.Append($" [{h.gameObject.name} active={h.isActiveAndEnabled} occupy={cols?.Length ?? -1}]");
            }
            return sb.ToString();
        }

        private static int CountPieces()
        {
            return FindObjectsByType<Games.Rebuilt.GamePieceSystem.RebuiltGamePieceController>(FindObjectsInactive.Exclude, FindObjectsSortMode.None).Length;
        }

        private string State(AttachProbe p)
        {
            var gm = BaseGameManager.Instance;
            var robot = p != null ? p.gameObject : null;
            Vector3 pos = robot ? robot.transform.localPosition : Vector3.zero;
            var body = robot ? robot.GetComponent<Rigidbody>() : null;
            return $"t={gm?.Timer:F2} gs={gm?.GameState} rs={gm?.RobotState} cheated={gm?.wasCheated} " +
                   $"blue={RebuiltScoreUI.TotalBlueScore} red={RebuiltScoreUI.TotalRedScore} " +
                   $"hub={RebuiltShifts.ActiveHub} shift={RebuiltShifts.ShiftNumber} " +
                   $"pos=({pos.x:F2},{pos.z:F2}) v={(body ? body.velocity.magnitude : 0f):F2} " +
                   $"fixedSteps={p?.FixedCount} exc={_excTotal} excRT={_excRollerThunker} ts={Time.timeScale}";
        }

        private IEnumerator MatchProbe()
        {
            Log.Write("match probe: waiting for spawned blue robot");
            float deadline = Time.realtimeSinceStartup + 45f;
            GameObject robot = null;
            while (Time.realtimeSinceStartup < deadline && (robot = FindBlueRobot()) == null)
                yield return null;
            if (robot == null)
            {
                Log.Write("P0 FAIL: no blue robot spawned within 45 s");
                Finish(false);
                yield break;
            }

            var probe = Attach(robot);
            LogEnv("match");

            var gm = BaseGameManager.Instance;
            var tagsField = typeof(BaseGameManager).GetField("tagsToDestroy", BindingFlags.Instance | BindingFlags.NonPublic);
            var tags = tagsField?.GetValue(gm) as string[];
            Log.Write($"gameManager={gm?.GetType().FullName} tagsToDestroy=[{(tags == null ? "?" : string.Join(",", tags))}]");
            Log.Write(HubReport() + $" pieces={CountPieces()}");

            // P0 proof: FixedUpdate must actually tick on the attached component.
            int before = probe.FixedCount;
            yield return new WaitForSecondsRealtime(1.5f);
            bool p0 = probe.FixedCount > before;
            Log.Write($"P0 {(p0 ? "PASS" : "FAIL")}: AttachProbe.FixedUpdate ticks {before}->{probe.FixedCount} on '{robot.name}'");
            _summary["p0_pass"] = p0;
            _summary["robot"] = robot.name;
            _summary["team"] = robot.GetComponent<RobotBase>()?.TeamNumber ?? -1;

            // Drive test: field-oriented half-speed +X for 1 s, then stop. Requires RobotState.Enabled.
            if (Entry.Flag("DRIVE_TEST"))
            {
                float wait = Time.realtimeSinceStartup + 20f;
                while (Time.realtimeSinceStartup < wait && BaseGameManager.Instance.RobotState != MoSimCore.Enums.RobotState.Enabled)
                    yield return null;
                Vector3 p0pos = robot.transform.localPosition;
                probe.DriveCmd = new Vector2(0.5f, 0f);
                probe.DriveUntil = Time.time + 1f;
                yield return new WaitForSeconds(1.5f);
                Vector3 p1pos = robot.transform.localPosition;
                probe.DriveCmd = new Vector2(0f, 0.5f);
                probe.DriveUntil = Time.time + 1f;
                yield return new WaitForSeconds(1.5f);
                Vector3 p2pos = robot.transform.localPosition;
                Log.Write($"drive: cmd(+x) moved ({p1pos.x - p0pos.x:F2},{p1pos.z - p0pos.z:F2}); " +
                          $"cmd(+y) moved ({p2pos.x - p1pos.x:F2},{p2pos.z - p1pos.z:F2})  [local x,z metres]");
            }

            // Button injection: can we press 4414's own buttons through its MoSimInput wrappers?
            if (Entry.Flag("BUTTON_TEST"))
            {
                ButtonInjector inj = null;
                try
                {
                    inj = ButtonInjector.Install(robot.GetComponent<Games.Rebuilt.Robots.RebuiltRobotBase>());
                    Log.Write($"buttons: installed; readback idle intake={inj.ReadBack(Button.Intake)}");
                }
                catch (Exception e)
                {
                    Log.Write("buttons: install FAILED: " + e);
                }
                if (inj != null)
                {
                    inj.Set(Button.Intake, true);
                    yield return new WaitForSeconds(0.2f);
                    bool held = inj.ReadBack(Button.Intake);
                    int blue0 = RebuiltScoreUI.TotalBlueScore;
                    yield return new WaitForSeconds(2f);
                    inj.Set(Button.Intake, false);
                    inj.Set(Button.AutoShoot, true);
                    yield return new WaitForSeconds(0.2f);
                    bool shootHeld = inj.ReadBack(Button.AutoShoot);
                    yield return new WaitForSeconds(3f);
                    inj.ReleaseAll();
                    yield return new WaitForSeconds(0.2f);
                    Log.Write($"buttons: intake pressed readback={held}; autoshoot pressed readback={shootHeld}; " +
                              $"released readback={inj.ReadBack(Button.AutoShoot)}; blue {blue0}->{RebuiltScoreUI.TotalBlueScore} during 3 s AutoShoot");
                    _summary["buttons_ok"] = held && shootHeld;
                    inj.Dispose();
                }
            }

            for (int i = 0; i < 8; i++)
            {
                Log.Write("state " + State(probe));
                yield return new WaitForSecondsRealtime(1f);
            }

            // ResetMatch drift test.
            int resets = Entry.Flag("RESET_TEST") ? 3 : 0;
            for (int r = 0; r < resets; r++)
            {
                Log.Write($"reset#{r} before: {HubReport()} pieces={CountPieces()} robotId={robot.GetInstanceID()} blue={RebuiltScoreUI.TotalBlueScore}");
                float t0 = Time.realtimeSinceStartup;
                gm.StartCoroutine(gm.ResetMatch());
                yield return null;
                while (gm.IsResetting) yield return null;
                float dt = Time.realtimeSinceStartup - t0;
                // Non-zero here = stale scorers surviving the reset (reward double-count across episodes).
                Log.Write($"reset#{r} immediately after: blue={RebuiltScoreUI.TotalBlueScore} autoFuel={RebuiltScoreUI.blueAutoFuelScore}");
                yield return new WaitForSecondsRealtime(2f);
                var robot2 = FindBlueRobot();
                if (robot2 == null)
                {
                    Log.Write($"reset#{r}: no blue robot after reset — stopping reset test");
                    break;
                }
                bool sameRobot = robot2 == robot;
                if (!sameRobot || probe == null) { robot = robot2; probe = Attach(robot2); }
                Log.Write($"reset#{r} after {dt:F2}s: {HubReport()} pieces={CountPieces()} robotId={robot2?.GetInstanceID()} " +
                          $"sameRobot={sameRobot} blue={RebuiltScoreUI.TotalBlueScore} cheated={gm.wasCheated} t={gm.Timer:F1}");
            }

            // timeScale throughput test: fixed steps & match-seconds per wall-second.
            if (Entry.Flag("TIMESCALE_TEST"))
            {
                foreach (float ts in new[] { 1f, 2f, 4f, 8f })
                {
                    Time.timeScale = ts;
                    int s0 = probe.FixedCount; float m0 = gm.Timer; float w0 = Time.realtimeSinceStartup;
                    int e0 = _excRollerThunker; int f0 = Time.frameCount;
                    yield return new WaitForSecondsRealtime(4f);
                    float w = Time.realtimeSinceStartup - w0;
                    Log.Write($"timescale {ts}x: fixedSteps/wall-s={(probe.FixedCount - s0) / w:F1} " +
                              $"match-s/wall-s={(m0 - gm.Timer) / w:F2} frames/wall-s={(Time.frameCount - f0) / w:F1} " +
                              $"rtExc/wall-s={(_excRollerThunker - e0) / w:F1} cheated={gm.wasCheated} gs={gm.GameState}");
                    _summary[$"ts{ts}_matchSecPerWallSec"] = (m0 - gm.Timer) / w;
                    _summary[$"ts{ts}_fixedStepsPerWallSec"] = (probe.FixedCount - s0) / w;
                }
                Time.timeScale = 1f;
                gm.StartCoroutine(gm.ResetMatch());
                yield return null;
                while (gm.IsResetting) yield return null;
                Log.Write($"post-timescale reset done: t={gm.Timer:F1} ts={Time.timeScale}");
            }

            Finish(p0);
        }

        private void Finish(bool p0)
        {
            _summary["exceptions_total"] = _excTotal;
            _summary["exceptions_rollerthunker"] = _excRollerThunker;
            var sb = new StringBuilder("{");
            sb.Append(string.Join(",", _summary.Select(kv =>
                $"\"{kv.Key}\":{(kv.Value is string s ? "\"" + s + "\"" : kv.Value is bool b ? (b ? "true" : "false") : Convert.ToString(kv.Value, System.Globalization.CultureInfo.InvariantCulture))}")));
            sb.Append("}");
            try { File.WriteAllText(Path.Combine(Entry.RunDir, "probe-summary.json"), sb.ToString()); } catch { }
            Log.Write("probe finished: " + sb);
            if (Entry.Flag("AUTOQUIT"))
            {
                Log.Write("autoquit");
                Application.Quit();
            }
        }
    }

    public class AttachProbe : MonoBehaviour
    {
        public int FixedCount;
        public Vector2 DriveCmd;
        public float DriveUntil;
        private DriveController _drive;

        private void Awake() => _drive = GetComponent<DriveController>();

        private void FixedUpdate()
        {
            FixedCount++;
            if (_drive != null && Time.time < DriveUntil)
                _drive.overideInput(DriveCmd, 0f, DriveController.DriveMode.FieldOriented);
        }
    }
}
