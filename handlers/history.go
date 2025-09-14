package handlers

import (
	"hack/database"
	"hack/models"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

func GetAllAnalyses(c *gin.Context) {
	var analyses []models.Analysis

	limitStr := c.DefaultQuery("limit", "50")
	offsetStr := c.DefaultQuery("offset", "0")

	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit <= 0 {
		limit = 50
	}

	offset, err := strconv.Atoi(offsetStr)
	if err != nil || offset < 0 {
		offset = 0
	}

	result := database.DB.Order("created_at DESC").Limit(limit).Offset(offset).Find(&analyses)
	if result.Error != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch analyses"})
		return
	}

	var total int64
	database.DB.Model(&models.Analysis{}).Count(&total)

	c.JSON(http.StatusOK, gin.H{
		"data":   analyses,
		"total":  total,
		"limit":  limit,
		"offset": offset,
	})
}

func GetAnalysisById(c *gin.Context) {
	id := c.Param("id")

	var analysis models.Analysis
	result := database.DB.First(&analysis, "id = ?", id)

	if result.Error != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Analysis not found"})
		return
	}

	c.JSON(http.StatusOK, analysis)
}

func DeleteAnalysis(c *gin.Context) {
	id := c.Param("id")

	result := database.DB.Delete(&models.Analysis{}, "id = ?", id)

	if result.Error != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete analysis"})
		return
	}

	if result.RowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "Analysis not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Analysis deleted successfully"})
}

func GetStatistics(c *gin.Context) {
	var stats struct {
		TotalAnalyses  int64   `json:"total_analyses"`
		WithRust       int64   `json:"with_rust"`
		WithCracks     int64   `json:"with_cracks"`
		WithDirt       int64   `json:"with_dirt"`
		AvgCleanliness float64 `json:"avg_cleanliness"`
	}

	database.DB.Model(&models.Analysis{}).Count(&stats.TotalAnalyses)
	database.DB.Model(&models.Analysis{}).Where("has_rust = ?", true).Count(&stats.WithRust)
	database.DB.Model(&models.Analysis{}).Where("has_cracks = ?", true).Count(&stats.WithCracks)
	database.DB.Model(&models.Analysis{}).Where("has_dirt = ?", true).Count(&stats.WithDirt)

	var avgCleanliness *float64
	database.DB.Model(&models.Analysis{}).Select("AVG(cleanliness_score)").Scan(&avgCleanliness)
	if avgCleanliness != nil {
		stats.AvgCleanliness = *avgCleanliness
	}

	c.JSON(http.StatusOK, stats)
}
