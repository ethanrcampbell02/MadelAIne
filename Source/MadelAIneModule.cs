using Celeste;
using Microsoft.Xna.Framework;
using Monocle;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Text.Json;

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
    private Vector2 LastExactPos = Vector2.Zero;
    private string LastRoomName = null;
    private bool PlayerDied = false;
    private bool ReachedNextRoom = false;
    private List<int[]> SolidTileData = null;
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

        // UDP client removed for SendGameState
        // TCP client will be created in SendGameState as needed
        On.Monocle.Engine.Update += Engine_Update;
    }

    public override void Unload()
    {
        On.Monocle.Engine.Update -= Engine_Update;
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

    private void Engine_Update(On.Monocle.Engine.orig_Update orig, Engine self, GameTime gameTime)
    {
        orig(self, gameTime);

        if (Engine.Scene is Level level && Settings.EnableMadelAIne && !level.Paused)
        {
            Player player = level.Tracker.GetEntity<Player>();
            UpdateState(player, level);
        }
    }

    private void UpdateState(Player player, Level level)
    {
        if (player == null)
        {
            if (LastPlayer == null) return;
            player = LastPlayer;
        }
        LastPlayer = player;

        Vector2 pos = player.ExactPosition;

        Vector2 velocity = Vector2.Zero;
        if (LastExactPos != Vector2.Zero)
        {
            velocity = pos - LastExactPos;
        }

        LastExactPos = pos;

        // Check if player died during this step
        bool playerDiedThisStep = false;
        if (player.Dead && !PlayerDied)
        {
            PlayerDied = true;
            playerDiedThisStep = true;
        }

        UpdateRoomLayout();

        // Update the game state
        GameState state = new GameState
        {
            PlayerXPosition = pos.X,
            PlayerYPosition = pos.Y,
            PlayerXVelocity = velocity.X,
            PlayerYVelocity = velocity.Y,
            PlayerCanDash = player.CanDash,
            PlayerStamina = player.Stamina,
            PlayerDied = playerDiedThisStep,
            NextRoom = ReachedNextRoom,
            TargetXPosition = 264f,               // FIXME: Currently hardcoded to second room of 1A
            TargetYPosition = -24f,               // FIXME: Currently hardcoded to second room of 1A
            SolidTileData = SolidTileData
        };

        SendGameState(state);
    }

    private void UpdateRoomLayout()
    {
        if (!(Engine.Scene is Level)) return;
        Level level = (Level)Engine.Scene;

        string debugRoomName = level.Session.Level;
        //If we are in the same room as last frame
        if (LastRoomName != null && LastRoomName == debugRoomName) return;
        if (LastRoomName != null && LastRoomName != debugRoomName)
        {
            ReachedNextRoom = true;
        }
        
        LastRoomName = debugRoomName;

        Logger.Info(nameof(MadelAIneModule), $"Entered new room: '{debugRoomName}'");

        int offsetX = level.LevelSolidOffset.X;
        int offsetY = level.LevelSolidOffset.Y;
        int width = (int)Math.Ceiling(level.Bounds.Width / 8f);
        int height = (int)Math.Ceiling(level.Bounds.Height / 8f);
        SolidTileData = new List<int[]>();
        for (int y = 0; y < height; y++)
        {
            int[] row = new int[width];
            string line = "";
            for (int x = 0; x < width; x++)
            {
                row[x] = level.SolidTiles.Grid.Data[x + offsetX, y + offsetY] ? 1 : 0;
                line += row[x] == 1 ? "1" : " ";
            }
            SolidTileData.Add(row);
        }
    }

    private void ResetGameState()
    {
        LastPlayer = null;
        LastExactPos = Vector2.Zero;
        PlayerDied = false;
        ReachedNextRoom = false;
        LastRoomName = null;
        SolidTileData = new List<int[]>();

        // Load Celeste/1-ForsakenCity room 1
        if (!(Engine.Scene is Level)) return;
        Level level = (Level)Engine.Scene;
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null)
        {
            // Player is currently dead, just reload the room
            level.Reload();
        }
        else
        {
            // Player is alive, reset back to room 1
            level.TeleportTo(player, "1", Player.IntroTypes.Respawn);
        }
    }
    private void SendGameState(GameState state)
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

            bool reset = ReceiveResponse();

            if (reset)
            {
                ResetGameState();
            }
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
        }
    }

    private bool ReceiveResponse()
    {
        Logger.Debug(nameof(MadelAIneModule), "Waiting for ACK...");
        byte[] buffer = new byte[1024];
        int bytesRead = tcpStream.Read(buffer, 0, buffer.Length);
        string response = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim('\0');
        try
        {
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
        catch (JsonException)
        {
            Logger.Error(nameof(MadelAIneModule), $"Invalid JSON response from Python client: {response}");
            return false;
        }
    }
}