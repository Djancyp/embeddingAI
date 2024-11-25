package db

import (
	"database/sql"
	"fmt"
	"log"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

var Database *sql.DB

// ConnectSql initializes the database and connects to it
func ConnectSql() {
	// Initialize the SQLite Vector bindings
	sqlite_vec.Auto()

	// Connect to an in-memory SQLite database
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		log.Fatal("Error opening database:", err)
	}
	Database = db

	// Retrieve and print SQLite and Vector extension versions
	var sqliteVersion, vecVersion string
	err = db.QueryRow("SELECT sqlite_version(), vec_version()").Scan(&sqliteVersion, &vecVersion)
	if err != nil {
		log.Fatal("Error fetching versions:", err)
	}
	fmt.Printf("SQLite Version: %s, Vec Version: %s\n", sqliteVersion, vecVersion)

	// Create a virtual table to store both text and embeddings
	_, err = db.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
		rowid TEXT PRIMARY KEY,
		embedding FLOAT[768]
)`)
	if err != nil {
		log.Fatal("Error creating virtual table:", err)
	}
	// create another table to store the TEXT
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS text_items (
   rowid TEXT PRIMARY KEY,
   text text
   )`)
	if err != nil {
		log.Fatal("Error creating virtual table:", err)
	}
	// create index for text_items and vec_items
	_, err = db.Exec(`CREATE INDEX IF NOT EXISTS text_items_index ON text_items(rowid)`)
	if err != nil {
		log.Fatal("Error creating index:", err)
	}
}

// Insert inserts text and its corresponding embedding into the vec_items table
func Insert(id string, embedding []float32, text string) error {
	// Serialize the embedding
	InserErr := InsertText(id, text)
	if InserErr != nil {
		return fmt.Errorf("Error inserting into text_items: %v", InserErr)
	}
	v, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("Error serializing vector: %v", err)
	}

	// Insert both the text and its embedding
	_, err = Database.Exec("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", id, v)
	if err != nil {
		return fmt.Errorf("Error inserting into vec_items: %v", err)
	}

	return nil
}

func InsertText(id string, text string) error {
	_, err := Database.Exec("INSERT INTO text_items(rowid, text) VALUES (?, ?)", id, text)
	if err != nil {
		return fmt.Errorf("Error inserting into text_items: %v", err)
	}
	return nil
}

type QueryResult struct {
	RowID    string
	Text     string
	Distance float64
}

// Query performs a vector search in the vec_items table and returns the matched items
func Query(queryVector []float32) ([]QueryResult, error) {
	// Serialize the query vector for matching
	queryResults := []QueryResult{}
	query, err := sqlite_vec.SerializeFloat32(queryVector)
	if err != nil {
		return queryResults, fmt.Errorf("Error serializing query vector: %v", err)
	}

	// Perform a vector search in the vec_items table
	rows, err := Database.Query(`
		SELECT
			rowid,
			distance
		FROM vec_items
		WHERE embedding MATCH ?
		ORDER BY distance
		LIMIT 3
	`, query)
	if err != nil {
		return queryResults, fmt.Errorf("Error querying vec_items: %v", err)
	}
	defer rows.Close()

	// Iterate through query results and print rowid, text, and distance
	for rows.Next() {
		q := QueryResult{}
		err = rows.Scan(&q.RowID, &q.Distance)
		if err != nil {
			return queryResults, fmt.Errorf("Error scanning row: %v", err)
		}
		queryResults = append(queryResults, q)
	}

	// Check for any errors encountered during row iteration
	if err = rows.Err(); err != nil {
		return queryResults, fmt.Errorf("Error during row iteration: %v", err)
	}

	// get text from text_items
	for i, item := range queryResults {
		text, err := GetText(item.RowID)
		if err != nil {
			return queryResults, fmt.Errorf("Error fetching text: %v", err)
		}
		queryResults[i].Text = text
	}

	return queryResults, nil
}

func GetText(rowid string) (string, error) {
	var text string
	err := Database.QueryRow("SELECT text FROM text_items WHERE rowid = ?", rowid).Scan(&text)
	if err != nil {
		return "", fmt.Errorf("Error fetching text: %v", err)
	}
	return text, nil
}

func FlushAllData() error {
	_, err := Database.Exec("DELETE FROM vec_items")
	if err != nil {
		return fmt.Errorf("Error deleting from vec_items: %v", err)
	}
	_, err = Database.Exec("DELETE FROM text_items")
	if err != nil {
		return fmt.Errorf("Error deleting from text_items: %v", err)
	}
	return nil
}
