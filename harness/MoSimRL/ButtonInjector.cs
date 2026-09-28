// Presses 4414's driver buttons programmatically without touching the human's devices.
//
// RebuiltRobotBase exposes IntakeAction / OuttakeAction / AutoPass / ManualShoot / RobotSpecial as public
// MoSimInput getters; MoSimInput.IsPressed() just returns its private `action.IsPressed()`. We swap that
// private InputAction for one bound to a private virtual gamepad, so the robot's own logic (turret aim,
// tilt compensation, feed gating) runs exactly as for a human driver.

using System;
using System.Collections.Generic;
using System.Reflection;
using Games.Rebuilt.Robots;
using MoSimLib;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;

namespace MoSimRL
{
    public enum Button { Intake, AutoShoot, AutoPass, ManualShoot, RobotSpecial }

    public sealed class ButtonInjector : IDisposable
    {
        private static readonly FieldInfo ActionField =
            typeof(MoSimInput).GetField("action", BindingFlags.Instance | BindingFlags.NonPublic);

        private static readonly (Button button, string property, GamepadButton pad)[] Map =
        {
            (Button.Intake, nameof(RebuiltRobotBase.IntakeAction), GamepadButton.South),
            (Button.AutoShoot, nameof(RebuiltRobotBase.OuttakeAction), GamepadButton.East),
            (Button.AutoPass, nameof(RebuiltRobotBase.AutoPass), GamepadButton.West),
            (Button.ManualShoot, nameof(RebuiltRobotBase.ManualShoot), GamepadButton.North),
            (Button.RobotSpecial, nameof(RebuiltRobotBase.RobotSpecial), GamepadButton.LeftShoulder),
        };

        private readonly Gamepad _pad;
        private readonly List<(MoSimInput input, InputAction original, InputAction injected)> _swaps =
            new List<(MoSimInput, InputAction, InputAction)>();
        private readonly Dictionary<Button, MoSimInput> _inputs = new Dictionary<Button, MoSimInput>();
        private GamepadState _state;
        // Rotate stick → turret feed-forward (Hightide: turret.Align(RotateAction.ReadValue<float>()), FF = value × 6 rad/s).
        // A human's rotate stick also feeds the turret; a scripted driver (overideInput) leaves it at 0. We swap the
        // robot's RotateAction for one bound to this pad's right stick X so the bot's rotation can feed the turret too.
        private static readonly FieldInfo RotateField =
            typeof(RobotFramework.RobotBase).GetField("<RotateAction>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic);
        private RobotFramework.RobotBase _robot;
        private InputAction _origRotate, _injRotate;

        private ButtonInjector(Gamepad pad) => _pad = pad;

        public static ButtonInjector Install(RebuiltRobotBase robot)
        {
            if (ActionField == null) throw new MissingFieldException("MoSimInput.action");
            var pad = InputSystem.AddDevice<Gamepad>("MoSimRLPad");
            var inj = new ButtonInjector(pad);
            foreach (var (button, property, padButton) in Map)
            {
                var prop = typeof(RebuiltRobotBase).GetProperty(property, BindingFlags.Instance | BindingFlags.Public);
                var input = prop?.GetValue(robot) as MoSimInput;
                if (input == null) throw new InvalidOperationException($"robot has no MoSimInput '{property}'");
                var original = ActionField.GetValue(input) as InputAction;
                var control = (ButtonControl)pad[padButton];
                // Device-qualified path: binds to our virtual pad only, never to the user's controller.
                var injected = new InputAction("MoSimRL." + button, InputActionType.Button, control.path);
                injected.Enable();
                ActionField.SetValue(input, injected);
                inj._swaps.Add((input, original, injected));
                inj._inputs[button] = input;
            }
            if (RotateField != null)
            {
                inj._robot = robot;
                inj._origRotate = RotateField.GetValue(robot) as InputAction;
                inj._injRotate = new InputAction("MoSimRL.Rotate", InputActionType.Value, pad.rightStick.x.path);
                inj._injRotate.Enable();
                RotateField.SetValue(robot, inj._injRotate);
            }
            return inj;
        }

        public void Set(Button button, bool down)
        {
            var padButton = Map[(int)button].pad;
            _state = _state.WithButton(padButton, down);
            InputSystem.QueueStateEvent(_pad, _state);
        }

        public void SetAll(bool[] down, float rotate = 0f)
        {
            var st = new GamepadState();
            for (int i = 0; i < Map.Length && i < down.Length; i++)
                if (down[i]) st = st.WithButton(Map[i].pad, true);
            st.rightStick = new UnityEngine.Vector2(UnityEngine.Mathf.Clamp(rotate, -1f, 1f), 0f);
            _state = st;
            InputSystem.QueueStateEvent(_pad, _state);
        }

        public float RotateReadBack() => _robot != null && _robot.RotateAction != null ? _robot.RotateAction.ReadValue<float>() : 0f;

        public void ReleaseAll()
        {
            _state = new GamepadState();
            InputSystem.QueueStateEvent(_pad, _state);
        }

        public bool ReadBack(Button button) => _inputs[button].IsPressed();

        public void Dispose()
        {
            foreach (var (input, original, injected) in _swaps)
            {
                ActionField.SetValue(input, original);
                injected.Disable();
                injected.Dispose();
            }
            _swaps.Clear();
            if (_robot != null && RotateField != null && _origRotate != null) RotateField.SetValue(_robot, _origRotate);
            if (_injRotate != null) { _injRotate.Disable(); _injRotate.Dispose(); _injRotate = null; }
            if (_pad != null && _pad.added) InputSystem.RemoveDevice(_pad);
        }
    }
}
