package main

import (
	"hack/database"
	"hack/handlers"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	database.InitDatabase()

	router := gin.Default()

	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	})

	router.MaxMultipartMemory = 10 << 20

	api := router.Group("/api")
	{
		api.POST("/analyze", handlers.UploadAndAnalyze)

		api.GET("/history", handlers.GetAllAnalyses)
		api.GET("/history/:id", handlers.GetAnalysisById)
		api.DELETE("/history/:id", handlers.DeleteAnalysis)

		api.GET("/statistics", handlers.GetStatistics)
	}

	router.Static("/uploads", "./uploads")
	router.Static("/static", "./static")

	// Serve frontend
	router.GET("/", func(c *gin.Context) {
		c.File("./static/index.html")
	})

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"service": "Image Analysis API",
		})
	})

	port := ":8081"
	log.Printf("Server starting on port %s", port)
	if err := router.Run(port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}