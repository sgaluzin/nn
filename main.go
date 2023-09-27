package main

import (
	"encoding/csv"
	"fmt"
	. "gorgonia.org/gorgonia"
	"log"
	"math/rand"
	"os"
	"strconv"
)

var err error

func main() {
	rand.Seed(0)

	// Create graph and network
	g := NewGraph()
	nn := NewNN(g)

	// Set input x to network
	//xB := getInputData()

	xB, yB := getInputAndValidationData("datasets/data.csv")

	x := nn.CreateInputMatrix(xB)

	// Define validation data set
	//yB := getValidationData()
	y := nn.CreateValidationMatrix(yB)

	// Run forward pass
	if err = nn.Fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	cost := nn.CalculateCost(y)

	// Do Gradient updates
	if _, err = Grad(cost, nn.Learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	tapeMachine := NewTapeMachine(g, BindDualValues(nn.Learnables()...))
	solver := NewVanillaSolver(WithLearnRate(LEARN_RATE))

	for i := 0; i < ITER_COUNT; i++ {
		if err = tapeMachine.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}

		err := solver.Step(NodesToValueGrads(nn.Learnables()))
		if err != nil {
			panic(err)
		}

		accuracyValue := nn.Accuracy(y.Value().Data().([]float64))

		if i%1000 == 0 {
			fmt.Printf(
				"Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
				i,
				cost.Value(),
				accuracyValue)
		}

		if 0.03 > cost.Value().Data().(float64) {
			break
		}

		tapeMachine.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", nn.predVal)

	// Set input x to network
	//testB := []float64{0, 0, 0, 0, 0, 0, 0, 1}
	//
	//testT := tensor.New(tensor.WithBacking(testB), tensor.WithShape(4, 2))
	//
	//testx := NewMatrix(g,
	//	tensor.Float64,
	//	WithName("testX"),
	//	WithShape(4, 2),
	//	WithValue(testT),
	//)
	//
	//err := nn.Fwd(testx)
	//if err != nil {
	//	panic(err)
	//}
	//
	//tapeMachine1 := NewTapeMachine(g, BindDualValues(nn.Learnables()...))
	//
	//if err = tapeMachine1.RunAll(); err != nil {
	//	panic(err)
	//}
	//
	//fmt.Println(nn.predVal.Data())
}

func getInputAndValidationData(fileName string) ([]float64, []float64) {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// read csv values using csv.Reader
	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var inputs []float64
	var outputs []float64
	var v float64
	for i := 0; i < len(data); i++ {
		if i == 0 {
			continue
		}
		for j := 0; j < 8; j++ {
			v, _ = strconv.ParseFloat(data[i][j], 64)
			inputs = append(inputs, v)
		}
		for k := 8; k < 8+5; k++ {
			v, _ = strconv.ParseFloat(data[i][k], 64)
			outputs = append(outputs, v)
		}
	}

	return inputs, outputs
}

func getInputData() []float64 {
	xB := []float64{1, 0, 0, 1, 1, 1, 0, 0}
	return xB
}

func getValidationData() []float64 {
	yB := []float64{1, 1, 0, 0}
	return yB
}
