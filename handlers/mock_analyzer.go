package handlers

import (
	"hack/models"
	"math/rand"
	"time"
)

func mockAnalyzeImage(imagePath string) (*models.ModelResponse, error) {
	rand.Seed(time.Now().UnixNano())

	hasRust := rand.Float32() < 0.3
	hasCracks := rand.Float32() < 0.2
	hasDirt := rand.Float32() < 0.4
	cleanliness := rand.Float32()*100

	status := "good"
	description := "Vehicle appears to be in good condition"

	if hasRust || hasCracks {
		status = "needs_attention"
		description = "Vehicle shows signs of wear and requires attention"
	}

	if hasRust && hasCracks && cleanliness < 50 {
		status = "poor"
		description = "Vehicle is in poor condition and needs immediate maintenance"
	}

	if cleanliness > 80 && !hasRust && !hasCracks {
		status = "excellent"
		description = "Vehicle is in excellent condition"
	}

	return &models.ModelResponse{
		Rust:        hasRust,
		Cracks:      hasCracks,
		Dirt:        hasDirt,
		Cleanliness: float64(cleanliness),
		Status:      status,
		Description: description,
	}, nil
}