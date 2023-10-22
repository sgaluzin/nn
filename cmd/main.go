package main

import (
	"bufio"
	"fmt"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"os"
	"strconv"
	"strings"
	nn2 "try2/nn"
)

var err error

func main() {
	inputsAmount := 5
	nn := nn2.LoadNN("../nn.bin")

	x := gorgonia.NewMatrix(nn.GetGraph(),
		tensor.Float64,
		gorgonia.WithName("X"),
		gorgonia.WithShape(1, inputsAmount),
	)
	nn.Fwd(x)

	machine := gorgonia.NewTapeMachine(nn.GetGraph())
	defer machine.Close()

	values := make([]float64, inputsAmount)
	for {
		values[0] = getInput("UserId") / 1000
		values[1] = getInput("Gender")
		values[2] = getInput("Age") / 100.0
		values[3] = getInput("AnnualSalary") / 100000.0
		values[4] = 1
		xT := tensor.New(tensor.WithBacking(values), tensor.WithShape(1, inputsAmount))

		machine.Let(x, xT)
		if err = machine.RunAll(); err != nil {
			log.Fatal(err)
		}
		machine.Reset()

		fmt.Println(values, nn.PredVal)
	}
}

func getInput(s string) float64 {
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("%v: ", s)
	text, _ := reader.ReadString('\n')
	text = strings.TrimSpace(text)

	input, err := strconv.ParseFloat(text, 64)
	if err != nil {
		log.Fatal(err)
	}
	return input
}
