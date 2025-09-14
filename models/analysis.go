package models

import (
	"time"
)

type DamageDetail struct {
	Type       string  `json:"type"`        // "scratch", "dent", "rust", etc.
	Confidence float64 `json:"confidence"`  // 0.0 - 1.0
	X          float64 `json:"x"`          // X coordinate (center)
	Y          float64 `json:"y"`          // Y coordinate (center)
	Width      float64 `json:"width"`      // Bounding box width
	Height     float64 `json:"height"`     // Bounding box height
	Area       float64 `json:"area"`       // Calculated area
	Severity   string  `json:"severity"`   // "minor", "moderate", "severe"
	Description string `json:"description"` // Human-readable description
	DetectedBy string  `json:"detected_by"` // Which model detected this
}

type Analysis struct {
	ID              string         `json:"id" gorm:"primaryKey"`
	ImagePath       string         `json:"image_path"`
	OriginalName    string         `json:"original_name"`
	HasRust         bool           `json:"has_rust"`
	HasCracks       bool           `json:"has_cracks"`
	HasDirt         bool           `json:"has_dirt"`
	HasScratches    bool           `json:"has_scratches"`
	HasDents        bool           `json:"has_dents"`
	CleanlinessScore float64       `json:"cleanliness_score"`
	OverallStatus   string         `json:"overall_status"`
	Details         string         `json:"details"`
	DetectionCounts map[string]int `json:"detection_counts" gorm:"serializer:json"`
	ConfidenceScores map[string]float64 `json:"confidence_scores" gorm:"serializer:json"`
	DamageDetails   []DamageDetail `json:"damage_details" gorm:"serializer:json"`
	ModelComparison *ModelComparison `json:"model_comparison,omitempty" gorm:"serializer:json"`
	CreatedAt       time.Time      `json:"created_at"`
	UpdatedAt       time.Time      `json:"updated_at"`
}

type AnalysisRequest struct {
	ImageBase64 string `json:"image_base64,omitempty"`
}

type ModelComparison struct {
	Local    ModelResponse `json:"local"`    // Local models response
	Offline  ModelResponse `json:"offline"`
	Agreement map[string]bool `json:"agreement"` // Which detections both methods agree on
	Conflicts []string `json:"conflicts"` // Where methods disagree
	Combined  ModelResponse `json:"combined"` // Final combined result
}

type ModelResponse struct {
	Rust         bool    `json:"rust"`
	Cracks       bool    `json:"cracks"`
	Dirt         bool    `json:"dirt"`
	Scratches    bool    `json:"scratches"`
	Dents        bool    `json:"dents"`
	Cleanliness  float64 `json:"cleanliness"`
	Status       string  `json:"status"`
	Description  string  `json:"description"`
	DetectionCounts map[string]int `json:"detection_counts,omitempty"`
	ConfidenceScores map[string]float64 `json:"confidence_scores,omitempty"`
	DamageDetails []DamageDetail `json:"damage_details,omitempty"`
	Method       string  `json:"method,omitempty"` // "local", "offline", "combined", "yolo"
}