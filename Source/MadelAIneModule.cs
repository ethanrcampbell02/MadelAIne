using Celeste;
using Microsoft.Xna.Framework;
using Monocle;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.Xna.Framework.Graphics;
using DeepCopy;
using Celeste.Mod.ProgrammaticInput.Binds;

/* Note: Several code snippets used to calculate the current state have been adapted
   from viddie's Physics Inspector in the Consistency Tracker mod.

   This is because Cyber is a lazy bum :3
*/


namespace Celeste.Mod.MadelAIne;

public class MadelAIneModule : EverestModule
{
    public static MadelAIneModule Instance { get; private set; }

    public override Type SettingsType => typeof(MadelAIneModuleSettings);
    public static MadelAIneModuleSettings Settings => (MadelAIneModuleSettings)Instance._Settings;

    public override Type SessionType => typeof(MadelAIneModuleSession);
    public static MadelAIneModuleSession Session => (MadelAIneModuleSession)Instance._Session;

    public override Type SaveDataType => typeof(MadelAIneModuleSaveData);
    public static MadelAIneModuleSaveData SaveData => (MadelAIneModuleSaveData)Instance._SaveData;

    private Player LastPlayer = null;
    private string LastRoomName = null;
    private bool ResetRequested = false;
    private TcpClient tcpClient = null;
    private NetworkStream tcpStream = null;
    private bool lastCalledWasUpdate = false;
    private Session savedSession = null;


    public MadelAIneModule()
    {
        Instance = this;
        Logger.SetLogLevel(nameof(MadelAIneModule), LogLevel.Info);
    }

    public override void Load()
    {
        Logger.Log(nameof(MadelAIneModule), $"Loading MadelAIne");
        On.Monocle.Engine.Draw += Engine_Draw;
        On.Monocle.Engine.Update += Engine_Update;
        On.Celeste.Level.Render += Level_Render;
    }

    public override void Unload()
    {
        On.Celeste.Level.Render -= Level_Render;
        On.Monocle.Engine.Update -= Engine_Update;
        On.Monocle.Engine.Draw -= Engine_Draw;
        
        // Reset all inputs before unloading
        ResetInputs();
        
        if (tcpStream != null)
        {
            tcpStream.Close();
            tcpStream = null;
        }
        if (tcpClient != null)
        {
            tcpClient.Close();
            tcpClient = null;
        }
    }

    public void Engine_Update(On.Monocle.Engine.orig_Update orig, Engine self, GameTime gameTime) {
        if (!Settings.EnableMadelAIne) {
            orig(self, gameTime);
            return;
        }

        if (!lastCalledWasUpdate) {
            // Handle cutscenes - skip them automatically
            if (Engine.Scene is Level level && level.InCutscene) {
                if (!level.SkippingCutscene) {
                    Logger.Info(nameof(MadelAIneModule), "Skipping cutscene...");
                    level.SkipCutscene();
                }
                orig(self, gameTime);
                lastCalledWasUpdate = true;
                return;
            }

            // Execute pending reset when level transition completes
            if (ResetRequested && Engine.Scene is Level resetLevel && !resetLevel.Transitioning) {
                Player player = resetLevel.Tracker.GetEntity<Player>();
                if (player != null && !player.Dead) {
                    Logger.Info(nameof(MadelAIneModule), "Resetting game state...");
                    ResetGameState();
                    ResetRequested = false;
                }
            }

            orig(self, gameTime);
            lastCalledWasUpdate = true;
        }
    }

    private void Engine_Draw(On.Monocle.Engine.orig_Draw orig, Monocle.Engine engine, GameTime gameTime)
    {
        if (!Settings.EnableMadelAIne) {
            orig(engine, gameTime);
            return;
        }

        if (lastCalledWasUpdate) {
            orig(engine, gameTime);
            lastCalledWasUpdate = false;
        }
    }

    private void Level_Render(On.Celeste.Level.orig_Render orig, Level self)
    {
        orig(self);

        if (!Settings.EnableMadelAIne || self.Paused || ResetRequested) return;

        // Save the session on first render when MadelAIne is enabled and in a level
        if (savedSession == null)
        {
            SaveSession(self);
        }

        Player player = self.Tracker.GetEntity<Player>();

        // Update the game state with the player's position and other relevant data
        GameState state = GetState(player, self);
        if (state == null) return;

        // Send the game state to the server
        bool success = SendGameState(state);
        if (!success) return;

        // Receive the response from the server, reset state if requested.
        // Note: Reset will be executed in Engine_Update when transition completes
        bool resetReceived = ReceiveResponse();
        if (resetReceived)
        {
            ResetRequested = true;
        }
    }

    private GameState GetState(Player player, Level level)
    {
        if (level == null) return null;

        if (player == null) {
            if (LastPlayer == null) return null;
            player = LastPlayer;
        }
        LastPlayer = player;

        Vector2 pos = player.ExactPosition;

        string debugRoomName = level.Session.Level;
        bool reachedNextRoom = LastRoomName != null && LastRoomName != debugRoomName;
        LastRoomName = debugRoomName;

        RenderTarget2D target = GameplayBuffers.Level.Target;
        Color[] pixels = new Color[target.Width * target.Height];
        target.GetData(pixels);
        byte[] pixelBytes = new byte[pixels.Length * 4];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixelBytes[i * 4 + 0] = pixels[i].R;
            pixelBytes[i * 4 + 1] = pixels[i].G;
            pixelBytes[i * 4 + 2] = pixels[i].B;
            pixelBytes[i * 4 + 3] = pixels[i].A;
        }
        string base64String = Convert.ToBase64String(pixelBytes);

        GameState state = new GameState
        {
            PlayerXPosition = pos.X,
            PlayerYPosition = pos.Y,
            PlayerDied = player.Dead,
            PlayerReachedNextRoom = reachedNextRoom,
            TargetXPosition = 2000f,                // FIXME: Currently hardcoded to final dash cutscene of prologue
            TargetYPosition = 60f,                  // FIXME: Currently hardcoded to final dash cutscene of prologue
            ScreenWidth = target.Width,
            ScreenHeight = target.Height,
            ScreenPixelsBase64 = base64String,
            LevelName = debugRoomName
        };

        return state;
    }

    private T DeepCopy<T>(T obj)
    {
        if (obj == null) return default(T);
        return DeepCopier.Copy(obj);
    }

    private void SaveSession(Level level)
    {
        if (level?.Session == null) return;

        try
        {
            savedSession = DeepCopy(level.Session);
            Logger.Info(nameof(MadelAIneModule), $"Saved session state for level: {savedSession.Level}");
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error saving session: {ex.Message}");
            savedSession = null;
        }
    }

    private void RestoreSession(Level level)
    {
        if (savedSession == null || level?.Session == null) return;

        try
        {
            var restoredSession = DeepCopy(savedSession);
            // Copy the restored session properties to the current session
            level.Session.Level = restoredSession.Level;
            level.Session.RespawnPoint = restoredSession.RespawnPoint;
            level.Session.Inventory = restoredSession.Inventory;
            level.Session.Flags = restoredSession.Flags;
            level.Session.LevelFlags = restoredSession.LevelFlags;
            level.Session.Strawberries = restoredSession.Strawberries;
            level.Session.DoNotLoad = restoredSession.DoNotLoad;
            level.Session.Keys = restoredSession.Keys;
            level.Session.Counters = restoredSession.Counters;
            level.Session.FurthestSeenLevel = restoredSession.FurthestSeenLevel;
            level.Session.StartCheckpoint = restoredSession.StartCheckpoint;
            level.Session.ColorGrade = restoredSession.ColorGrade;
            level.Session.SummitGems = restoredSession.SummitGems;
            level.Session.FirstLevel = restoredSession.FirstLevel;
            level.Session.Cassette = restoredSession.Cassette;
            level.Session.HeartGem = restoredSession.HeartGem;
            level.Session.Dreaming = restoredSession.Dreaming;
            level.Session.GrabbedGolden = restoredSession.GrabbedGolden;
            level.Session.HitCheckpoint = restoredSession.HitCheckpoint;
            
            Logger.Info(nameof(MadelAIneModule), $"Restored session state for level: {level.Session.Level}");
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error restoring session: {ex.Message}");
        }
    }

    private void ResetGameState()
    {

        if (!(Engine.Scene is Level)) return;
        Level level = (Level)Engine.Scene;
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null)
        {
            if (LastPlayer == null) return;
            player = LastPlayer;
        }
        
        // Reset all inputs before resetting game state
        ResetInputs();
        
        // Restore the saved session if available
        if (savedSession != null)
        {
            RestoreSession(level);
            level.TeleportTo(player, savedSession.Level, Player.IntroTypes.Respawn);
        }

        LastPlayer = null;
        LastRoomName = null;
    }
    
    private void ResetInputs()
    {
        // Reset all GameplayBinds to neutral/released state
        GameplayBinds.MoveX.SetNeutral();
        GameplayBinds.MoveY.SetNeutral();
        
        GameplayBinds.Jump.Release();
        GameplayBinds.Dash.Release();
        GameplayBinds.Grab.Release();
        GameplayBinds.Talk.Release();
        GameplayBinds.CrouchDash.Release();
        
        Logger.Debug(nameof(MadelAIneModule), "Reset all GameplayBinds inputs");
    }
    
    private bool SendGameState(GameState state)
    {
        // Send the game state to the Python client using TCP
        string json = JsonSerializer.Serialize(state);
        byte[] data = Encoding.UTF8.GetBytes(json);

        try
        {
            // Ensure connection is open and healthy
            if (tcpClient == null || !tcpClient.Connected)
            {
                if (tcpStream != null)
                {
                    tcpStream.Close();
                    tcpStream = null;
                }
                if (tcpClient != null)
                {
                    tcpClient.Close();
                    tcpClient = null;
                }
                var localEndPoint = new IPEndPoint(IPAddress.Loopback, 5001);
                tcpClient = new TcpClient();
                tcpClient.NoDelay = true;
                tcpClient.Client.Bind(localEndPoint);
                tcpClient.Connect("127.0.0.1", 5000);
                tcpStream = tcpClient.GetStream();
            }

            tcpStream.Write(data, 0, data.Length);

            return true;
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error sending game state: {ex.Message}");
            CleanupConnection();
            return false;
        }
    }

    private bool ReceiveResponse()
    {
        Logger.Debug(nameof(MadelAIneModule), "Waiting for ACK...");
        try
        {
            byte[] buffer = new byte[1024];
            int bytesRead = tcpStream.Read(buffer, 0, buffer.Length);
            
            // Check if connection was closed gracefully
            if (bytesRead == 0)
            {
                Logger.Info(nameof(MadelAIneModule), "Python client closed connection gracefully");
                CleanupConnection();
                return false;
            }
            
            string response = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim('\0');

            using var doc = JsonDocument.Parse(response);
            if (!doc.RootElement.TryGetProperty("type", out var typeProp))
            {
                Logger.Error(nameof(MadelAIneModule), $"Unexpected response from Python client: {response}");
                return false;
            }
            string type = typeProp.GetString();
            if (type == "ACK")
            {
                // Parse and apply inputs from ACK message
                ApplyInputs(doc.RootElement);
                return false;
            }
            else if (type == "reset")
            {
                Logger.Info(nameof(MadelAIneModule), "Reset requested by Python client.");
                return true;
            }
            else if (type == "shutdown")
            {
                Logger.Info(nameof(MadelAIneModule), "Shutdown requested by Python client");
                CleanupConnection();
                return false;
            }
            else
            {
                Logger.Error(nameof(MadelAIneModule), $"Unexpected response type from Python client: {type}");
                return false;
            }
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error receiving response from Python client: {ex.Message}");
            CleanupConnection();
            return false;
        }
    }
    
    private void CleanupConnection()
    {
        // Clean up connection and state
        if (tcpStream != null)
        {
            tcpStream.Close();
            tcpStream = null;
        }
        if (tcpClient != null)
        {
            tcpClient.Close();
            tcpClient = null;
        }
        
        savedSession = null;
        ResetRequested = false;
        ResetInputs();
        Settings.EnableMadelAIne = false;
        
        Logger.Info(nameof(MadelAIneModule), "Connection cleaned up, MadelAIne disabled");
    }

    private void ApplyInputs(JsonElement response)
    {
        try
        {
            // Parse input values from response
            // Expected format: {"type": "ACK", "moveX": 0.0, "moveY": 0.0, "jump": false, "dash": false, "grab": false}
            
            // Movement axes
            if (response.TryGetProperty("moveX", out var moveXProp))
            {
                float moveX = moveXProp.GetSingle();
                if (moveX == 0)
                    GameplayBinds.MoveX.SetNeutral();
                else
                    GameplayBinds.MoveX.SetValue(moveX);
            }
            
            if (response.TryGetProperty("moveY", out var moveYProp))
            {
                float moveY = moveYProp.GetSingle();
                if (moveY == 0)
                    GameplayBinds.MoveY.SetNeutral();
                else
                    GameplayBinds.MoveY.SetValue(moveY);
            }
            
            // Button presses
            if (response.TryGetProperty("jump", out var jumpProp))
            {
                if (jumpProp.GetBoolean())
                    GameplayBinds.Jump.Press();
                else
                    GameplayBinds.Jump.Release();
            }
            
            if (response.TryGetProperty("dash", out var dashProp))
            {
                if (dashProp.GetBoolean())
                    GameplayBinds.Dash.Press();
                else
                    GameplayBinds.Dash.Release();
            }
            
            if (response.TryGetProperty("grab", out var grabProp))
            {
                if (grabProp.GetBoolean())
                    GameplayBinds.Grab.Press();
                else
                    GameplayBinds.Grab.Release();
            }
            
            Logger.Debug(nameof(MadelAIneModule), "Applied inputs from Python");
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error applying inputs: {ex.Message}");
        }
    }
}