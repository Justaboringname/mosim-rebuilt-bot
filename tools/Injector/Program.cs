// Adds a guarded bootstrap to the user's own MoSim mod DLL (China Modpack/Alphabots.dll) so that when
// Unity constructs the mod's robot MonoBehaviour during mod loading, MoSimRL.dll is loaded and
// MoSimRL.Entry.Init() is called. Everything else in the assembly is untouched.
// Install/restore: tools/install-hook.sh, tools/restore-hook.sh (restore = copy the backup back).
//
// usage: Injector <in.dll> <out.dll> <gameManagedDir> <harnessDllAbsPath> <hostTypeFullName>
using System;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

static class Program
{
    const string BootTypeName = "__MoSimRLBoot";

    static int Main(string[] a)
    {
        if (a.Length != 5) { Console.Error.WriteLine("usage: Injector <in.dll> <out.dll> <managedDir> <harnessDll> <hostType>"); return 2; }
        string input = a[0], output = a[1], managed = a[2], harness = a[3], hostName = a[4];

        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(managed);
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(input)));
        var rp = new ReaderParameters { AssemblyResolver = resolver, ReadingMode = ReadingMode.Immediate, InMemory = true };

        using var asm = AssemblyDefinition.ReadAssembly(input, rp);
        var mod = asm.MainModule;
        Console.WriteLine($"assembly: {asm.Name.FullName}");
        foreach (var r in mod.AssemblyReferences) Console.WriteLine($"  ref: {r.FullName}");

        if (mod.Types.Any(t => t.Name == BootTypeName)) { Console.Error.WriteLine("already injected"); return 3; }
        var host = mod.GetType(hostName) ?? throw new Exception("host type not found: " + hostName);

        // Import BCL/Unity members from the GAME's assemblies, never from this .NET 9 process.
        var corlib = AssemblyDefinition.ReadAssembly(Path.Combine(managed, "mscorlib.dll"), rp).MainModule;
        var unity = AssemblyDefinition.ReadAssembly(Path.Combine(managed, "UnityEngine.CoreModule.dll"), rp).MainModule;
        TypeDefinition T(ModuleDefinition m, string n) => m.GetType(n) ?? throw new Exception("missing type " + n);
        MethodReference M(TypeDefinition t, string name, params string[] ps) =>
            mod.ImportReference(t.Methods.First(x => x.Name == name && x.Parameters.Count == ps.Length &&
                x.Parameters.Select(p => p.ParameterType.FullName).SequenceEqual(ps)));

        var fileExists = M(T(corlib, "System.IO.File"), "Exists", "System.String");
        var readAll = M(T(corlib, "System.IO.File"), "ReadAllBytes", "System.String");
        var asmLoad = M(T(corlib, "System.Reflection.Assembly"), "Load", "System.Byte[]");
        var asmGetType = M(T(corlib, "System.Reflection.Assembly"), "GetType", "System.String");
        var typeGetMethod = M(T(corlib, "System.Type"), "GetMethod", "System.String");
        var invoke = M(T(corlib, "System.Reflection.MethodBase"), "Invoke", "System.Object", "System.Object[]");
        var toStr = M(T(corlib, "System.Object"), "ToString");
        var concat = M(T(corlib, "System.String"), "Concat", "System.String", "System.String");
        var logWarn = M(T(unity, "UnityEngine.Debug"), "LogWarning", "System.Object");
        var exType = mod.ImportReference(T(corlib, "System.Exception"));

        var boot = new TypeDefinition("", BootTypeName,
            TypeAttributes.NotPublic | TypeAttributes.Abstract | TypeAttributes.Sealed | TypeAttributes.Class,
            mod.TypeSystem.Object);
        var done = new FieldDefinition("done", FieldAttributes.Private | FieldAttributes.Static, mod.TypeSystem.Boolean);
        boot.Fields.Add(done);
        var run = new MethodDefinition("Run", MethodAttributes.Public | MethodAttributes.Static | MethodAttributes.HideBySig, mod.TypeSystem.Void);
        boot.Methods.Add(run);
        mod.Types.Add(boot);

        // static void Run() {
        //   try { if (done) return; done = true;
        //         if (File.Exists(H)) Assembly.Load(File.ReadAllBytes(H)).GetType("MoSimRL.Entry").GetMethod("Init").Invoke(null, null); }
        //   catch (Exception ex) { Debug.LogWarning("[MoSimRL] boot failed: " + ex); }
        // }
        var body = run.Body;
        var exLocal = new VariableDefinition(exType);
        body.Variables.Add(exLocal);
        body.InitLocals = true;
        var il = body.GetILProcessor();
        var end = il.Create(OpCodes.Ret);
        var leaveTry = il.Create(OpCodes.Leave, end);

        var tryStart = il.Create(OpCodes.Ldsfld, done);
        il.Append(tryStart);
        il.Append(il.Create(OpCodes.Brtrue, leaveTry));
        il.Append(il.Create(OpCodes.Ldc_I4_1));
        il.Append(il.Create(OpCodes.Stsfld, done));
        il.Append(il.Create(OpCodes.Ldstr, harness));
        il.Append(il.Create(OpCodes.Call, fileExists));
        il.Append(il.Create(OpCodes.Brfalse, leaveTry));
        il.Append(il.Create(OpCodes.Ldstr, harness));
        il.Append(il.Create(OpCodes.Call, readAll));
        il.Append(il.Create(OpCodes.Call, asmLoad));
        il.Append(il.Create(OpCodes.Ldstr, "MoSimRL.Entry"));
        il.Append(il.Create(OpCodes.Callvirt, asmGetType));
        il.Append(il.Create(OpCodes.Ldstr, "Init"));
        il.Append(il.Create(OpCodes.Callvirt, typeGetMethod));
        il.Append(il.Create(OpCodes.Ldnull));
        il.Append(il.Create(OpCodes.Ldnull));
        il.Append(il.Create(OpCodes.Callvirt, invoke));
        il.Append(il.Create(OpCodes.Pop));
        il.Append(leaveTry);
        var catchStart = il.Create(OpCodes.Stloc, exLocal);
        il.Append(catchStart);
        il.Append(il.Create(OpCodes.Ldstr, "[MoSimRL] boot failed: "));
        il.Append(il.Create(OpCodes.Ldloc, exLocal));
        il.Append(il.Create(OpCodes.Callvirt, toStr));
        il.Append(il.Create(OpCodes.Call, concat));
        il.Append(il.Create(OpCodes.Call, logWarn));
        il.Append(il.Create(OpCodes.Leave, end));
        il.Append(end);
        body.ExceptionHandlers.Add(new ExceptionHandler(ExceptionHandlerType.Catch)
        {
            TryStart = tryStart, TryEnd = catchStart,
            HandlerStart = catchStart, HandlerEnd = end,
            CatchType = exType,
        });

        int patched = 0;
        foreach (var ctor in host.Methods.Where(m => m.IsConstructor && !m.IsStatic && m.HasBody))
        {
            var cil = ctor.Body.GetILProcessor();
            cil.InsertBefore(ctor.Body.Instructions[0], cil.Create(OpCodes.Call, run));
            patched++;
        }
        if (patched == 0) throw new Exception("no instance ctor on host type");
        Console.WriteLine($"patched {patched} ctor(s) on {host.FullName}; harness = {harness}");

        asm.Write(output);
        Console.WriteLine("wrote " + output);
        return 0;
    }
}
