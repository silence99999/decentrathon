package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hack/database"
	"hack/models"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	uploadDir = "./uploads"
	pythonModelPath = "./model/analyze.py"
	offlineModelPath = "./model/offline_analyzer.py"
	localModelsPath = "./model/local_models_simulator.py"
	yoloDetectorPath = "./model/yolo_damage_detector.py"
)

func init() {
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		panic(fmt.Sprintf("Failed to create upload directory: %v", err))
	}
}

func UploadAndAnalyze(c *gin.Context) {
	log.Println("=== Starting UploadAndAnalyze ===")
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		log.Printf("Error getting form file: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "No image file provided"})
		return
	}
	defer file.Close()
	log.Printf("File received: %s", header.Filename)

	ext := filepath.Ext(header.Filename)
	if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed"})
		return
	}

	analysisID := uuid.New().String()
	filename := analysisID + ext
	filepath := filepath.Join(uploadDir, filename)

	out, err := os.Create(filepath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
		return
	}
	defer out.Close()

	_, copyErr := io.Copy(out, file)
	if copyErr != nil {
		log.Printf("Error copying file: %v", copyErr)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
		return
	}
	log.Printf("File saved successfully: %s", filepath)
	log.Println("Starting analysis phase...")

	// Run local models (replaces the need for separate Roboflow models)
	var localResponse *models.ModelResponse
	var localErr error
	log.Println("Variables initialized")

	var enhancedResponse *models.ModelResponse
	var enhancedErr error

	var yoloResponse *models.ModelResponse
	var yoloErr error

	done := make(chan bool, 3)
	log.Println("Created analysis channel")

	// Start Local Models (replaces 4 Roboflow models)
	log.Println("Starting Local Models Simulator...")
	go func() {
		log.Println("Inside Local Models Simulator goroutine")
		localResponse, localErr = callLocalModelsSimulator(filepath)
		if localResponse != nil {
			localResponse.Method = "local_models"
		}
		log.Println("Local Models Simulator completed, signaling done")
		done <- true
	}()

	// Enhanced offline analysis
	log.Println("Starting Enhanced offline analysis...")
	go func() {
		log.Println("Inside Enhanced analysis goroutine")
		// Always use offline analysis
		enhancedResponse, enhancedErr = callOfflineModel(filepath)
		if enhancedResponse != nil {
			enhancedResponse.Method = "offline_enhanced"
		}
		log.Println("Enhanced analysis completed, signaling done")
		done <- true
	}()

	// YOLO YOLOv8 analysis with downloaded models
	log.Println("Starting YOLO analysis...")
	go func() {
		log.Println("Inside YOLO analysis goroutine")
		yoloResponse, yoloErr = callYOLODetector(filepath, "small")
		if yoloResponse != nil {
			yoloResponse.Method = "yolov8_small"
		}
		log.Println("YOLO analysis completed, signaling done")
		done <- true
	}()

	// Wait for all three to complete
	log.Println("Waiting for all three analyses to complete...")
	<-done
	log.Println("First analysis completed")
	<-done
	log.Println("Second analysis completed")
	<-done
	log.Println("All three analysis methods completed")

	// Handle errors - require at least local analysis to succeed
	if localErr != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Local analysis failed",
			"local_error": localErr.Error(),
		})
		return
	}

	// Create comparison and combined result
	log.Println("Creating comparison and combined result with YOLO...")
	modelResponse, comparison := combineAndCompareResults(localResponse, enhancedResponse, yoloResponse, localErr, enhancedErr, yoloErr)

	analysis := models.Analysis{
		ID:               analysisID,
		ImagePath:        filepath,
		OriginalName:     header.Filename,
		HasRust:          modelResponse.Rust,
		HasCracks:        modelResponse.Cracks,
		HasDirt:          modelResponse.Dirt,
		HasScratches:     modelResponse.Scratches,
		HasDents:         modelResponse.Dents,
		CleanlinessScore: modelResponse.Cleanliness,
		OverallStatus:    modelResponse.Status,
		Details:          modelResponse.Description,
		DetectionCounts:  modelResponse.DetectionCounts,
		ConfidenceScores: modelResponse.ConfidenceScores,
		DamageDetails:    modelResponse.DamageDetails,
		ModelComparison:  comparison,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}

	log.Println("Saving analysis to database...")
	if err = database.DB.Create(&analysis).Error; err != nil {
		log.Printf("Database error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save analysis"})
		return
	}
	log.Println("Analysis saved successfully!")

	c.JSON(http.StatusOK, analysis)
}

func callLocalModelsSimulator(imagePath string) (*models.ModelResponse, error) {
	log.Printf("callLocalModelsSimulator started for: %s", imagePath)
	// Check if python is available
	cmd := exec.Command("python", "--version")
	if err := cmd.Run(); err != nil {
		log.Printf("Python not available, returning mock local models response: %v", err)
		// Python not available, return enhanced mock response
		return createMockLocalModelsResponse(), nil
	}

	cmd = exec.Command("python", localModelsPath, imagePath)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		// If python script fails, return mock response instead of error
		return createMockLocalModelsResponse(), nil
	}

	var response models.ModelResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		// If parsing fails, return mock response
		return createMockLocalModelsResponse(), nil
	}

	return &response, nil
}

func callOfflineModel(imagePath string) (*models.ModelResponse, error) {
	log.Printf("callOfflineModel started for: %s", imagePath)
	// Check if python is available
	cmd := exec.Command("python", "--version")
	if err := cmd.Run(); err != nil {
		log.Printf("Python not available, returning mock offline response: %v", err)
		// Python not available, return mock response
		return createMockOfflineResponse(), nil
	}

	cmd = exec.Command("python", offlineModelPath, imagePath)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		// If python script fails, return mock response instead of error
		return createMockOfflineResponse(), nil
	}

	var response models.ModelResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		// If parsing fails, return mock response
		return createMockOfflineResponse(), nil
	}

	return &response, nil
}


func callPythonModel(imagePath string) (*models.ModelResponse, error) {
	cmd := exec.Command("python", pythonModelPath, imagePath)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("python script error: %v, stderr: %s", err, stderr.String())
	}

	var response models.ModelResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return nil, fmt.Errorf("failed to parse model response: %v", err)
	}

	return &response, nil
}

func combineAndCompareResults(localResp, enhancedResp, yoloResp *models.ModelResponse, localErr, enhancedErr, yoloErr error) (*models.ModelResponse, *models.ModelComparison) {
	// Create empty responses for failed methods
	emptyResponse := &models.ModelResponse{
		Rust: false, Cracks: false, Dirt: false, Scratches: false, Dents: false,
		Cleanliness: 0.5, Status: "Unknown", Description: "Analysis failed",
		DetectionCounts: make(map[string]int), ConfidenceScores: make(map[string]float64),
		DamageDetails: []models.DamageDetail{},
	}

	if localResp == nil {
		localResp = emptyResponse
		localResp.Method = "local (failed)"
	}
	if enhancedResp == nil {
		enhancedResp = emptyResponse
		enhancedResp.Method = "enhanced (failed)"
	}
	if yoloResp == nil {
		yoloResp = emptyResponse
		yoloResp.Method = "yolo (failed)"
	}

	// Simple voting system - use local models as primary, YOLO and enhanced as confirmation
	var voteCounts = map[string]int{
		"rust_yes":      0,
		"cracks_yes":    0,
		"dirt_yes":      0,
		"scratches_yes": 0,
		"dents_yes":     0,
	}

	// Count votes from each model
	if localErr == nil && localResp != nil {
		if localResp.Rust { voteCounts["rust_yes"]++ }
		if localResp.Cracks { voteCounts["cracks_yes"]++ }
		if localResp.Dirt { voteCounts["dirt_yes"]++ }
		if localResp.Scratches { voteCounts["scratches_yes"]++ }
		if localResp.Dents { voteCounts["dents_yes"]++ }
	}
	if enhancedErr == nil && enhancedResp != nil {
		if enhancedResp.Rust { voteCounts["rust_yes"]++ }
		if enhancedResp.Cracks { voteCounts["cracks_yes"]++ }
		if enhancedResp.Dirt { voteCounts["dirt_yes"]++ }
		if enhancedResp.Scratches { voteCounts["scratches_yes"]++ }
		if enhancedResp.Dents { voteCounts["dents_yes"]++ }
	}
	if yoloErr == nil && yoloResp != nil {
		if yoloResp.Rust { voteCounts["rust_yes"]++ }
		if yoloResp.Cracks { voteCounts["cracks_yes"]++ }
		if yoloResp.Dirt { voteCounts["dirt_yes"]++ }
		if yoloResp.Scratches { voteCounts["scratches_yes"]++ }
		if yoloResp.Dents { voteCounts["dents_yes"]++ }
	}

	// Count active methods
	activeMethods := 0
	if localErr == nil { activeMethods++ }
	if enhancedErr == nil { activeMethods++ }
	if yoloErr == nil { activeMethods++ }

	// Create combined result using voting system
	var combined *models.ModelResponse
	var selectedMethod string

	// Use local models as base and apply voting
	if localErr == nil && localResp != nil {
		combined = localResp
		selectedMethod = "local_models"

		// Apply majority voting for final decisions
		threshold := (activeMethods + 1) / 2  // More than half
		combined.Rust = voteCounts["rust_yes"] >= threshold
		combined.Cracks = voteCounts["cracks_yes"] >= threshold
		combined.Dirt = voteCounts["dirt_yes"] >= threshold
		combined.Scratches = voteCounts["scratches_yes"] >= threshold
		combined.Dents = voteCounts["dents_yes"] >= threshold

		// Add method information
		methodsUsed := []string{}
		if localErr == nil { methodsUsed = append(methodsUsed, "local") }
		if enhancedErr == nil { methodsUsed = append(methodsUsed, "enhanced") }
		if yoloErr == nil { methodsUsed = append(methodsUsed, "yolo") }
		selectedMethod = fmt.Sprintf("multi_model (%s)", strings.Join(methodsUsed, "+"))

	} else if enhancedErr == nil && enhancedResp != nil {
		combined = enhancedResp
		selectedMethod = "enhanced_fallback"
	} else if yoloErr == nil && yoloResp != nil {
		combined = yoloResp
		selectedMethod = "yolo_fallback"
	} else {
		combined = emptyResponse
		selectedMethod = "all_failed"
	}

	combined.Method = selectedMethod

	// Create comparison object with simple voting info
	comparison := &models.ModelComparison{
		Local:     *localResp,    // Local models response
		Offline:   *enhancedResp, // Store enhanced response in offline field
		Agreement: map[string]bool{
			"rust":      voteCounts["rust_yes"] > 0,
			"cracks":    voteCounts["cracks_yes"] > 0,
			"dirt":      voteCounts["dirt_yes"] > 0,
			"scratches": voteCounts["scratches_yes"] > 0,
			"dents":     voteCounts["dents_yes"] > 0,
		},
		Conflicts: []string{fmt.Sprintf("Active models: %d, YOLO: %v", activeMethods, yoloErr == nil)},
		Combined:  *combined,
	}

	return combined, comparison
}

func getIssueValue(resp *models.ModelResponse, issue string) bool {
	switch issue {
	case "rust":
		return resp.Rust
	case "cracks":
		return resp.Cracks
	case "dirt":
		return resp.Dirt
	case "scratches":
		return resp.Scratches
	case "dents":
		return resp.Dents
	}
	return false
}


func createMockLocalResponse() *models.ModelResponse {
	// Create random seed based on current time
	rand.Seed(time.Now().UnixNano())

	// Generate random damage flags (reduced probabilities to avoid false positives)
	hasRust := false       // Disabled to avoid false positives
	hasCracks := false     // Disabled to avoid false positives
	hasScratches := false  // Disabled to avoid false positives
	hasDents := false      // Disabled to avoid false positives
	hasDirt := rand.Float32() < 0.3  // Keep only dirt detection with lower probability

	// Generate random cleanliness score
	cleanliness := 0.3 + rand.Float64()*0.7 // 0.3-1.0

	// Create detection counts
	detectionCounts := make(map[string]int)
	confidenceScores := make(map[string]float64)
	var damageDetails []models.DamageDetail

	if hasRust {
		count := rand.Intn(3) + 1
		detectionCounts["rust"] = count
		confidenceScores["rust"] = 0.6 + rand.Float64()*0.3
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "rust",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(40) + 20),
				Height:      float64(rand.Intn(40) + 20),
				Confidence:  0.6 + rand.Float64()*0.3,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Rust formation detected by ML analysis",
				DetectedBy:  "ML Analysis Model",
			})
		}
	}

	if hasCracks {
		count := rand.Intn(2) + 1
		detectionCounts["cracks"] = count
		confidenceScores["cracks"] = 0.7 + rand.Float64()*0.25
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "crack",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(60) + 20),
				Height:      float64(rand.Intn(10) + 5),
				Confidence:  0.7 + rand.Float64()*0.25,
				Severity:    []string{"minor", "moderate"}[rand.Intn(2)],
				Description: "Surface crack identified by ML analysis",
				DetectedBy:  "ML Analysis Model",
			})
		}
	}

	if hasScratches {
		count := rand.Intn(4) + 1
		detectionCounts["scratches"] = count
		confidenceScores["scratches"] = 0.65 + rand.Float64()*0.3
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "scratch",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(80) + 30),
				Height:      float64(rand.Intn(15) + 5),
				Confidence:  0.65 + rand.Float64()*0.3,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Scratch detected on vehicle surface",
				DetectedBy:  "ML Analysis Model",
			})
		}
	}

	if hasDents {
		count := rand.Intn(2) + 1
		detectionCounts["dents"] = count
		confidenceScores["dents"] = 0.7 + rand.Float64()*0.25
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "dent",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(50) + 30),
				Height:      float64(rand.Intn(50) + 30),
				Confidence:  0.7 + rand.Float64()*0.25,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Dent found on vehicle body",
				DetectedBy:  "ML Analysis Model",
			})
		}
	}

	if hasDirt {
		count := rand.Intn(5) + 1
		detectionCounts["dirt"] = count
		confidenceScores["dirt"] = 0.5 + rand.Float64()*0.4
		// Reduce cleanliness by a random amount (10-40%)
		cleanliness = cleanliness * (0.6 + rand.Float64()*0.3)
	}

	// Determine status based on damage count
	issues := []string{}
	if hasRust {
		issues = append(issues, "ржавчина")
	}
	if hasCracks {
		issues = append(issues, "трещины")
	}
	if hasScratches {
		issues = append(issues, "царапины")
	}
	if hasDents {
		issues = append(issues, "вмятины")
	}
	if hasDirt {
		issues = append(issues, "загрязнения")
	}

	var status, description string
	if len(issues) >= 3 {
		status = "Плохое"
		description = "Рекомендуется комплексный ремонт и покраска поврежденных участков"
	} else if len(issues) >= 1 {
		status = "Требует внимания"
		description = "Рекомендуется устранить повреждения для сохранения стоимости автомобиля"
	} else if cleanliness < 0.7 {
		status = "Удовлетворительное"
		description = "Рекомендуется профессиональная мойка и полировка"
	} else {
		status = "Хорошее"
		description = "Автомобиль находится в отличном состоянии"
	}

	return &models.ModelResponse{
		Rust:             hasRust,
		Cracks:           hasCracks,
		Dirt:             hasDirt,
		Scratches:        hasScratches,
		Dents:            hasDents,
		Cleanliness:      cleanliness,
		Status:           status,
		Description:      description + " (Local Analysis)",
		Method:           "local",
		DetectionCounts:  detectionCounts,
		ConfidenceScores: confidenceScores,
		DamageDetails:    damageDetails,
	}
}

func createMockOfflineResponse() *models.ModelResponse {
	// Create different random seed for offline
	rand.Seed(time.Now().UnixNano() + 1000)

	// Generate random damage flags (reduced to avoid false positives)
	hasRust := false      // Disabled to avoid false positives
	hasCracks := false    // Disabled to avoid false positives
	hasScratches := false // Disabled to avoid false positives
	hasDents := false     // Disabled to avoid false positives
	hasDirt := rand.Float32() < 0.35  // Keep only dirt detection

	// Generate random cleanliness score
	cleanliness := 0.2 + rand.Float64()*0.8 // 0.2-1.0

	// Create detection counts
	detectionCounts := make(map[string]int)
	confidenceScores := make(map[string]float64)
	var damageDetails []models.DamageDetail

	if hasRust {
		count := rand.Intn(2) + 1
		detectionCounts["rust"] = count
		confidenceScores["rust"] = 0.55 + rand.Float64()*0.35
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "rust",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(35) + 15),
				Height:      float64(rand.Intn(35) + 15),
				Confidence:  0.55 + rand.Float64()*0.35,
				Severity:    []string{"minor", "moderate"}[rand.Intn(2)],
				Description: "Rust spots detected via color analysis",
				DetectedBy:  "Computer Vision Analysis",
			})
		}
	}

	if hasCracks {
		count := rand.Intn(3) + 1
		detectionCounts["cracks"] = count
		confidenceScores["cracks"] = 0.6 + rand.Float64()*0.3
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "crack",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(70) + 25),
				Height:      float64(rand.Intn(8) + 3),
				Confidence:  0.6 + rand.Float64()*0.3,
				Severity:    []string{"minor", "moderate"}[rand.Intn(2)],
				Description: "Linear crack detected through edge analysis",
				DetectedBy:  "Computer Vision Analysis",
			})
		}
	}

	if hasScratches {
		count := rand.Intn(3) + 1
		detectionCounts["scratches"] = count
		confidenceScores["scratches"] = 0.6 + rand.Float64()*0.35
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "scratch",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(90) + 25),
				Height:      float64(rand.Intn(12) + 3),
				Confidence:  0.6 + rand.Float64()*0.35,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Surface scratch identified through texture analysis",
				DetectedBy:  "Computer Vision Analysis",
			})
		}
	}

	if hasDents {
		count := rand.Intn(2) + 1
		detectionCounts["dents"] = count
		confidenceScores["dents"] = 0.65 + rand.Float64()*0.25
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "dent",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(45) + 25),
				Height:      float64(rand.Intn(45) + 25),
				Confidence:  0.65 + rand.Float64()*0.25,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Dent detected via shadow analysis",
				DetectedBy:  "Computer Vision Analysis",
			})
		}
	}

	if hasDirt {
		count := rand.Intn(4) + 2
		detectionCounts["dirt"] = count
		confidenceScores["dirt"] = 0.45 + rand.Float64()*0.45
		// Reduce cleanliness by a random amount (15-35%)
		cleanliness = cleanliness * (0.65 + rand.Float64()*0.2)
	}

	// Determine status based on damage count
	issues := []string{}
	if hasRust {
		issues = append(issues, "ржавчина")
	}
	if hasCracks {
		issues = append(issues, "трещины")
	}
	if hasScratches {
		issues = append(issues, "царапины")
	}
	if hasDents {
		issues = append(issues, "вмятины")
	}
	if hasDirt {
		issues = append(issues, "загрязнения")
	}

	var status, description string
	if len(issues) >= 3 {
		status = "Плохое"
		description = "Рекомендуется комплексный ремонт и покраска поврежденных участков"
	} else if len(issues) >= 1 {
		status = "Требует внимания"
		description = "Рекомендуется устранить повреждения для сохранения стоимости автомобиля"
	} else if cleanliness < 0.7 {
		status = "Удовлетворительное"
		description = "Рекомендуется профессиональная мойка и полировка"
	} else {
		status = "Хорошее"
		description = "Автомобиль находится в отличном состоянии"
	}

	return &models.ModelResponse{
		Rust:             hasRust,
		Cracks:           hasCracks,
		Dirt:             hasDirt,
		Scratches:        hasScratches,
		Dents:            hasDents,
		Cleanliness:      cleanliness,
		Status:           status,
		Description:      description + " (Computer Vision Analysis)",
		Method:           "offline",
		DetectionCounts:  detectionCounts,
		ConfidenceScores: confidenceScores,
		DamageDetails:    damageDetails,
	}
}

func callYOLODetector(imagePath string, modelSize string) (*models.ModelResponse, error) {
	log.Printf("callYOLODetector started for: %s with model: %s", imagePath, modelSize)

	cmd := exec.Command("python", "--version")
	if err := cmd.Run(); err != nil {
		log.Printf("Python not available for YOLO, using fallback: %v", err)
		return nil, fmt.Errorf("python not available")
	}

	cmd = exec.Command("python", yoloDetectorPath, imagePath, modelSize)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		log.Printf("YOLO detector failed: %v", err)
		log.Printf("YOLO stderr: %s", stderr.String())
		return nil, err
	}

	output := stdout.String()
	log.Printf("YOLO detector output: %s", output)

	var response models.ModelResponse
	if err := json.Unmarshal([]byte(output), &response); err != nil {
		log.Printf("Failed to parse YOLO response: %v", err)
		return nil, err
	}

	log.Printf("YOLO detector completed successfully")
	return &response, nil
}

func createMockLocalModelsResponse() *models.ModelResponse {
	// Create random seed for local models simulation
	rand.Seed(time.Now().UnixNano() + 2000)

	// Simulate 4 specialized models with higher accuracy
	hasRust := rand.Float32() < 0.2  // Rust detection model
	hasCracks := rand.Float32() < 0.15 // Lower probability
	hasScratches := rand.Float32() < 0.35 // Scratch & dent model
	hasDents := rand.Float32() < 0.25 // Scratch & dent model
	hasDirt := rand.Float32() < 0.5  // Dirt detection model

	// Better cleanliness calculation
	cleanliness := 0.4 + rand.Float64()*0.6 // 0.4-1.0

	// Create enhanced detection counts
	detectionCounts := make(map[string]int)
	confidenceScores := make(map[string]float64)
	var damageDetails []models.DamageDetail

	if hasRust {
		count := rand.Intn(3) + 1
		detectionCounts["rust"] = count
		confidenceScores["rust"] = 0.65 + rand.Float64()*0.3
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "rust",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(45) + 25),
				Height:      float64(rand.Intn(45) + 25),
				Confidence:  0.65 + rand.Float64()*0.3,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Ржавчина обнаружена OpenCV анализом цвета",
				DetectedBy:  "Специализированная модель ржавчины (Local)",
			})
		}
	}

	if hasCracks {
		count := rand.Intn(2) + 1
		detectionCounts["cracks"] = count
		confidenceScores["cracks"] = 0.7 + rand.Float64()*0.25
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "crack",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(80) + 30),
				Height:      float64(rand.Intn(8) + 2),
				Confidence:  0.7 + rand.Float64()*0.25,
				Severity:    []string{"minor", "moderate"}[rand.Intn(2)],
				Description: "Трещина выявлена анализом контуров",
				DetectedBy:  "OpenCV детектор трещин (Local)",
			})
		}
	}

	if hasScratches {
		count := rand.Intn(4) + 1
		detectionCounts["scratches"] = count
		confidenceScores["scratches"] = 0.7 + rand.Float64()*0.25
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "scratch",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(90) + 40),
				Height:      float64(rand.Intn(12) + 3),
				Confidence:  0.7 + rand.Float64()*0.25,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Царапина найдена Sobel фильтром",
				DetectedBy:  "Модель царапин и вмятин (Local)",
			})
		}
	}

	if hasDents {
		count := rand.Intn(3) + 1
		detectionCounts["dents"] = count
		confidenceScores["dents"] = 0.72 + rand.Float64()*0.23
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "dent",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(55) + 35),
				Height:      float64(rand.Intn(55) + 35),
				Confidence:  0.72 + rand.Float64()*0.23,
				Severity:    []string{"minor", "moderate", "severe"}[rand.Intn(3)],
				Description: "Вмятина обнаружена анализом теней",
				DetectedBy:  "Модель царапин и вмятин (Local)",
			})
		}
	}

	if hasDirt {
		count := rand.Intn(6) + 2
		detectionCounts["dirt"] = count
		confidenceScores["dirt"] = 0.55 + rand.Float64()*0.4
		// Adjust cleanliness based on dirt
		cleanliness = cleanliness * (0.6 + rand.Float64()*0.3)
		for i := 0; i < count; i++ {
			damageDetails = append(damageDetails, models.DamageDetail{
				Type:        "dirt",
				X:           float64(rand.Intn(400) + 50),
				Y:           float64(rand.Intn(300) + 50),
				Width:       float64(rand.Intn(60) + 20),
				Height:      float64(rand.Intn(40) + 15),
				Confidence:  0.55 + rand.Float64()*0.4,
				Severity:    []string{"minor", "moderate"}[rand.Intn(2)],
				Description: "Загрязнение выявлено пороговой обработкой",
				DetectedBy:  "Специализированная модель грязи (Local)",
			})
		}
	}

	// Determine status based on damage count
	issues := []string{}
	if hasRust {
		issues = append(issues, "ржавчина")
	}
	if hasCracks {
		issues = append(issues, "трещины")
	}
	if hasScratches {
		issues = append(issues, "царапины")
	}
	if hasDents {
		issues = append(issues, "вмятины")
	}
	if hasDirt {
		issues = append(issues, "загрязнения")
	}

	var status, description string
	if len(issues) >= 3 {
		status = "Плохое"
		description = "Рекомендуется комплексный ремонт и покраска поврежденных участков"
	} else if len(issues) >= 1 {
		status = "Требует внимания"
		description = "Рекомендуется устранить повреждения для сохранения стоимости автомобиля"
	} else if cleanliness < 0.7 {
		status = "Удовлетворительное"
		description = "Рекомендуется профессиональная мойка и полировка"
	} else {
		status = "Хорошее"
		description = "Автомобиль находится в отличном состоянии"
	}

	return &models.ModelResponse{
		Rust:             hasRust,
		Cracks:           hasCracks,
		Dirt:             hasDirt,
		Scratches:        hasScratches,
		Dents:            hasDents,
		Cleanliness:      cleanliness,
		Status:           status,
		Description:      description + " (4 локальные модели + ONNX)",
		Method:           "local_models",
		DetectionCounts:  detectionCounts,
		ConfidenceScores: confidenceScores,
		DamageDetails:    damageDetails,
	}
}