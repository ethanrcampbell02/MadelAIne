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


    public MadelAIneModule()
    {
        Instance = this;
        Logger.SetLogLevel(nameof(MadelAIneModule), LogLevel.Info);
    }

    public override void Load()
    {
        Logger.Log(nameof(MadelAIneModule), $"Loading MadelAIne");
        On.Celeste.Level.Render += Level_Render;
        On.Monocle.Engine.Update += Engine_Update;
    }

    public override void Unload()
    {
        On.Monocle.Engine.Update -= Engine_Update;
        On.Celeste.Level.Render -= Level_Render;
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
        orig(self, gameTime);

        if (Engine.Scene is Level level && !level.Transitioning && ResetRequested) {
            ResetGameState();
        }
    }

    private void Level_Render(On.Celeste.Level.orig_Render orig, Level self)
    {
        orig(self);

        if (!Settings.EnableMadelAIne || self.Paused || ResetRequested) return;

        Player player = self.Tracker.GetEntity<Player>();

        // Update the game state with the player's position and other relevant data
        GameState state = GetState(player, self);
        if (state == null) return;

        // Send the game state to the server
        bool success = SendGameState(state);
        if (!success) return;

        // Receive the response from the server, reset state if requested.
        ResetRequested = ReceiveResponse();

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
            TargetXPosition = 264f,               // FIXME: Currently hardcoded to second room of 1A
            TargetYPosition = -24f,               // FIXME: Currently hardcoded to second room of 1A
            ScreenWidth = target.Width,
            ScreenHeight = target.Height,
            ScreenPixelsBase64 = base64String
        };

        return state;
    }

    private void ResetGameState()
    {

        // Load Celeste/1-ForsakenCity room 1
        if (!(Engine.Scene is Level)) return;
        Level level = (Level)Engine.Scene;
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null)
        {
            if (LastPlayer == null) return;
            player = LastPlayer;
        }
        level.TeleportTo(player, "1", Player.IntroTypes.Respawn);

        LastPlayer = null;
        LastRoomName = null;
        ResetRequested = false;
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
            Logger.Error(nameof(MadelAIneModule), $"Disabling MadelAIne");
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
            Settings.EnableMadelAIne = false;

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
                return false;
            }
            else if (type == "reset")
            {
                Logger.Info(nameof(MadelAIneModule), "Reset requested by Python client.");
                return true;
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
            return false;
        }
    }
}