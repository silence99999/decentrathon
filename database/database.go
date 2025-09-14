package database

import (
	"hack/models"
	"log"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
)

var DB *gorm.DB

func InitDatabase() {
	var err error
	DB, err = gorm.Open(sqlite.Open("analysis.db"), &gorm.Config{})
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	err = DB.AutoMigrate(&models.Analysis{})
	if err != nil {
		log.Fatal("Failed to migrate database:", err)
	}

	log.Println("Database connected and migrated successfully")
}