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

        [JsonPropertyName("playerXVelocity")]
        public float PlayerXVelocity { get; set; }

        [JsonPropertyName("playerYVelocity")]
        public float PlayerYVelocity { get; set; }

        [JsonPropertyName("playerCanDash")]
        public bool PlayerCanDash { get; set; }

        [JsonPropertyName("playerStamina")]
        public float PlayerStamina { get; set; }

        [JsonPropertyName("playerDied")]
        public bool PlayerDied { get; set; }

        [JsonPropertyName("targetXPosition")]
        public float TargetXPosition { get; set; }

        [JsonPropertyName("targetYPosition")]
        public float TargetYPosition { get; set; }

        [JsonPropertyName("solidTileData")]
        public List<int[]> SolidTileData { get; set; }
    }
}