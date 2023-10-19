package main

import (
	"fmt"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math"
)

var dt tensor.Dtype = tensor.Float64

type NN struct {
	g      *gorgonia.ExprGraph
	w0, w1 *gorgonia.Node

	pred    *gorgonia.Node
	predVal gorgonia.Value
}

const IN_COUNT = 8
const L1_COUNT = 14
const OUT_COUNT = 5
const ITER_COUNT = 100000
const LEARN_RATE = 0.3

func NewNN(g *gorgonia.ExprGraph) *NN {
	//weight layers
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(IN_COUNT, L1_COUNT), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.Uniform(0, 1)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(L1_COUNT, OUT_COUNT), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.Uniform(0, 1)))
	return &NN{
		g:  g,
		w0: w0,
		w1: w1}
}

func (m *NN) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1}
}

func (m *NN) Fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node

	//set inputs to l0
	l0 = x

	//multiply l0 on first weights
	l0dot = gorgonia.Must(gorgonia.Mul(l0, m.w0))

	//use activation function
	l1 = gorgonia.Must(gorgonia.Sigmoid(l0dot))

	//multiply hidden layer on second weights
	l1dot = gorgonia.Must(gorgonia.Mul(l1, m.w1))

	//use activation function
	l2 = gorgonia.Must(gorgonia.Sigmoid(l1dot))

	m.pred = l2
	gorgonia.Read(m.pred, &m.predVal)

	return nil
}

func (m *NN) CalculateCost(y *gorgonia.Node) *gorgonia.Node {
	losses := gorgonia.Must(gorgonia.Sub(y, m.pred))
	square := gorgonia.Must(gorgonia.Square(losses))
	cost := gorgonia.Must(gorgonia.Mean(square))
	return cost
}

func (m *NN) Accuracy(y []float64) float64 {
	prediction := m.predVal.Data().([]float64)

	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}

	return ok / float64(len(y))
}

func (m *NN) CreateInputMatrix(xB []float64) *gorgonia.Node {
	xLen := len(xB) / IN_COUNT
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(xLen, IN_COUNT))
	fmt.Println(xT)

	return gorgonia.NewMatrix(m.g,
		tensor.Float64,
		gorgonia.WithName("X"),
		gorgonia.WithShape(xLen, IN_COUNT),
		gorgonia.WithValue(xT),
	)
}

func (m *NN) CreateValidationMatrix(yB []float64) *gorgonia.Node {
	yBLen := len(yB) / OUT_COUNT
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(yBLen, OUT_COUNT))
	fmt.Println(yT)
	y := gorgonia.NewMatrix(m.g,
		tensor.Float64,
		gorgonia.WithName("y"),
		gorgonia.WithShape(yBLen, OUT_COUNT),
		gorgonia.WithValue(yT),
	)
	return y
}
