package main

import (
	"log"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

func convertEmbeding(text string) ([]float32, error) {
	// Download and cache pretrained tokenizer (e.g., BERT)
	r := []float32{}
	configFile, err := tokenizer.CachedPath("bert-base-uncased", "tokenizer.json")
	if err != nil {
		return r, err
	}

	tk, err := pretrained.FromFile(configFile)
	if err != nil {
		return r, err
	}

	// Sample sentence to tokenize
	sentence := text
	enc, err := tk.EncodeSingle(sentence)
	if err != nil {
		log.Fatal(err)
	}

	// Print tokenized input

	// Load the ONNX runtime shared library
	ort.SetSharedLibraryPath("./onnxruntime.so")

	err = ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// Prepare input tensors
	// input_ids (BERT input)
	inputData := make([]int64, len(enc.Ids))
	for i, id := range enc.Ids {
		inputData[i] = int64(id)
	}
	inputShape := ort.NewShape(1, int64(len(inputData))) // Batch size = 1
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	// token_type_ids (typically zeros for single sentence input)
	tokenTypeData := make([]int64, len(enc.Ids)) // Set to 0 for all tokens
	tokenTypeTensor, err := ort.NewTensor(inputShape, tokenTypeData)
	if err != nil {
		log.Fatal(err)
	}
	defer tokenTypeTensor.Destroy()

	// attention_mask (1 for real tokens, 0 for padding)
	attentionMaskData := make([]int64, len(enc.Ids))
	for i := 0; i < len(attentionMaskData); i++ {
		attentionMaskData[i] = 1 // No padding tokens in this example
	}
	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMaskData)
	if err != nil {
		log.Fatal(err)
	}
	defer attentionMaskTensor.Destroy()

	// Prepare output tensor (this is where the model will write the output)
	// The output shape depends on the sequence length (can be len(enc.Ids))
	outputShape := ort.NewShape(1, int64(len(enc.Ids)), 768) // Output shape
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatal(err)
	}
	defer outputTensor.Destroy()

	// Initialize the ONNX session
	session, err := ort.NewAdvancedSession("model_quantized.onnx",
		[]string{"input_ids", "token_type_ids", "attention_mask"}, // All required inputs
		[]string{"last_hidden_state"},
		[]ort.Value{inputTensor, tokenTypeTensor, attentionMaskTensor}, // Provide all inputs
		[]ort.Value{outputTensor}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// Run the model
	err = session.Run()
	if err != nil {
		log.Fatal(err)
	}

	// Get the output tensor data
	outputData := outputTensor.GetData()
	if len(outputData) >= 768 {
		outputData = outputData[:768]
	}
	return outputData, nil
}
