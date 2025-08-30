namespace Celeste.Mod.MadelAIne
{
    using System.Collections.Generic;
    using System.Text.Json.Serialization;

    public class GameState
    {
        [JsonPropertyName("playerXPosition")]
        public float PlayerXPosition { get; set; }

        [JsonPropertyName("playerYPosition")]
        public float PlayerYPosition { get; set; }

        [JsonPropertyName("playerDied")]
        public bool PlayerDied { get; set; }

        [JsonPropertyName("playerReachedNextRoom")]
        public bool PlayerReachedNextRoom { get; set; }

        [JsonPropertyName("targetXPosition")]
        public float TargetXPosition { get; set; }

        [JsonPropertyName("targetYPosition")]
        public float TargetYPosition { get; set; }

        [JsonPropertyName("screenWidth")]
        public int ScreenWidth { get; set; }

        [JsonPropertyName("screenHeight")]
        public int ScreenHeight { get; set; }

        [JsonPropertyName("screenPixelsBase64")]
        public string ScreenPixelsBase64 { get; set; }
    }
}