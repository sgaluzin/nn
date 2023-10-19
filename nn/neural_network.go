package nn

import (
	"encoding/gob"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math"
	"os"
)

type NN struct {
	g          *gorgonia.ExprGraph
	w0, w1     *gorgonia.Node
	pred       *gorgonia.Node
	PredVal    gorgonia.Value
	Cost       gorgonia.Value
	CostSquare gorgonia.Value
}

func NewNN(inputCount, outputCount, hiddenCount int) *NN {
	g := gorgonia.NewGraph()

	//weight layers
	w0 := gorgonia.NewMatrix(
		g,
		gorgonia.Float64,
		gorgonia.WithShape(inputCount, hiddenCount+1),
		gorgonia.WithName("w0"),
		gorgonia.WithInit(gorgonia.Uniform(0, 1)),
	)

	w1 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(hiddenCount+1, outputCount),
		gorgonia.WithName("w1"),
		gorgonia.WithInit(gorgonia.Uniform(0, 1)),
	)

	return &NN{
		g:       g,
		w0:      w0,
		w1:      w1,
		pred:    nil,
		PredVal: nil,
	}
}

func LoadNN(filename string) *NN {
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	dec := gob.NewDecoder(f)

	var wData []gorgonia.Value
	err = dec.Decode(&wData)
	if err != nil {
		panic(err)
	}

	g := gorgonia.NewGraph()
	w0 := gorgonia.NodeFromAny(g, wData[0], gorgonia.WithName("w0"))

	w1 := gorgonia.NodeFromAny(g, wData[1], gorgonia.WithName("w1"))

	return &NN{
		g,
		w0,
		w1,
		nil,
		nil,
		nil,
		nil,
	}
}

func (nn NN) Save() error {
	f, err := os.Create("nn.bin")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)

	err = enc.Encode([]gorgonia.Value{nn.w0.Value(), nn.w1.Value()})
	if err != nil {
		return err
	}

	return nil
}

func (nn NN) GetGraph() *gorgonia.ExprGraph {
	return nn.g
}

func (nn *NN) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{nn.w0, nn.w1}
}

func (nn *NN) Fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node

	//set inputs to l0
	l0 = x

	//multiply l0 on first weights
	l0dot = gorgonia.Must(gorgonia.Mul(l0, nn.w0))

	//use activation function
	//l1 = gorgonia.Must(gorgonia.LeakyRelu(l0dot, 0.01))
	//l1 = gorgonia.Must(gorgonia.Sigmoid(l0DotBias))
	l1 = gorgonia.Must(gorgonia.Rectify(l0dot))
	//l1 = gorgonia.Must(gorgonia.Tanh(l0dot))

	//multiply hidden layer on second weights
	l1dot = gorgonia.Must(gorgonia.Mul(l1, nn.w1))

	//use activation function
	//l2 = gorgonia.Must(gorgonia.LeakyRelu(l1DotBias, 0.01))
	//l2 = gorgonia.Must(gorgonia.Tanh(l1DotBias))
	//l2 = gorgonia.Must(gorgonia.Rectify(l1DotBias))
	//gorgonia.Read(l1dot, &nn.CostSquare)
	l2 = gorgonia.Must(gorgonia.Sigmoid(l1dot))

	nn.pred = l2

	gorgonia.Read(nn.pred, &nn.PredVal)

	return nil
}

func (nn *NN) CalculateCost(y *gorgonia.Node) *gorgonia.Node {
	//losses := gorgonia.Must(gorgonia.HadamardProd(nn.pred, y))
	//lossesMean := gorgonia.Must(gorgonia.Mean(losses))
	//cost := gorgonia.Must(gorgonia.Neg(lossesMean))
	//gorgonia.Read(cost, &nn.Cost)

	logProb := gorgonia.Must(gorgonia.Log(nn.pred))
	fstTerm := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Neg(y)), logProb))
	one := gorgonia.NewConstant(1.0)
	oneMinusY := gorgonia.Must(gorgonia.Sub(one, y))
	logOneMinusProb := gorgonia.Must(gorgonia.Log(gorgonia.Must(gorgonia.Sub(one, nn.pred))))
	sndTerm := gorgonia.Must(gorgonia.HadamardProd(oneMinusY, logOneMinusProb))
	crossEntropy := gorgonia.Must(gorgonia.Sub(fstTerm, sndTerm))
	cost := gorgonia.Must(gorgonia.Mean(crossEntropy))
	gorgonia.Read(cost, &nn.Cost)

	//losses := gorgonia.Must(gorgonia.Sub(y, nn.pred))
	//square := gorgonia.Must(gorgonia.Square(losses))
	//cost := gorgonia.Must(gorgonia.Mean(square))
	//gorgonia.Read(cost, &nn.Cost)

	return cost
}

func (nn *NN) Accuracy(y []float64) float64 {
	prediction := nn.PredVal.Data().([]float64)

	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}

	return ok / float64(len(y))
}
