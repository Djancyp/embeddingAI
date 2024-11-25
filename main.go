package main

import (
	"fmt"
	"os"
	"strings"

	"test/db"

	"github.com/google/uuid"
)

func main() {
	db.ConnectSql()
	file, err := os.ReadFile("test.txt")
	if err != nil {
		panic(err)
	}

	// Split the text file into lines

	lines := strings.Split(string(file), "\n\n")
	

	for _, text := range lines {
		if text == "" {
			continue
		}
		id := uuid.New().String()
		list, err := convertEmbeding(text)
		if err != nil {
			panic(err)
		}
		db.Insert(id, list, text)
	}

	// Query
	searchText := "Andy Murray will join his"
	list, err := convertEmbeding(searchText)
	queryResults, err := db.Query(list)
	if err != nil {
		panic(err)
	}

	defer db.Database.Close()
	fmt.Println(queryResults)
}
