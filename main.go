package main

import (
	"database/sql"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/go-sql-driver/mysql"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"math/rand"
	"os"
	nn2 "try2/nn"
)

const ITER_COUNT = 50000
const LEARN_RATE = 0.001
const BATCH_SIZE = 1000

var err error

func main() {
	rand.Seed(0)

	inputM, outputM := getInputDataFromFile()

	nn := nn2.NewNN(inputM.Ncol(), outputM.Ncol(), 3)

	x := gorgonia.NewMatrix(nn.GetGraph(),
		tensor.Float64,
		gorgonia.WithName("X"),
		gorgonia.WithShape(BATCH_SIZE, inputM.Ncol()),
	)

	y := gorgonia.NewMatrix(nn.GetGraph(),
		tensor.Float64,
		gorgonia.WithName("Y"),
		gorgonia.WithShape(BATCH_SIZE, outputM.Ncol()),
	)

	if err = nn.Fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	cost := nn.CalculateCost(y)

	if _, err = gorgonia.Grad(cost, nn.Learnables()...); err != nil {
		log.Fatal(err)
	}

	tapeMachine := gorgonia.NewTapeMachine(
		nn.GetGraph(),
		gorgonia.BindDualValues(nn.Learnables()...),
	)
	solver := gorgonia.NewAdamSolver(
		gorgonia.WithLearnRate(LEARN_RATE),
		gorgonia.WithBatchSize(BATCH_SIZE),
		gorgonia.WithClip(40),
		gorgonia.WithL2Reg(0.001),
		gorgonia.WithL1Reg(0.001),
	)

	var currentBatchSize int
	for i := 0; i < ITER_COUNT; i++ {
		for j := 0; j < inputM.Nrow(); j += BATCH_SIZE {
			if j+BATCH_SIZE > inputM.Nrow() {
				currentBatchSize = inputM.Nrow() - j - 1
			} else {
				currentBatchSize = BATCH_SIZE - 1
			}
			rowInput := inputM.Subset(makeRange(j, j+currentBatchSize))
			rowOutput := outputM.Subset(makeRange(j, j+currentBatchSize))

			xT := tensor.FromMat64(mat.DenseCopyOf(&matrix{rowInput}))
			yT := tensor.FromMat64(mat.DenseCopyOf(&matrix{rowOutput}))

			if err = tapeMachine.Let(x, xT); err != nil {
				panic(err)
			}

			if err = tapeMachine.Let(y, yT); err != nil {
				panic(err)
			}

			if err = tapeMachine.RunAll(); err != nil {
				log.Fatalf("Failed at inter  %d: %v", i, err)
			}

			//os.WriteFile("simple_graph.dot", []byte(nn.GetGraph().ToDot()), 0644)

			err := solver.Step(gorgonia.NodesToValueGrads(nn.Learnables()))
			if err != nil {
				panic(err)
			}

			tapeMachine.Reset()
		}

		accuracyValue := nn.Accuracy(y.Value().Data().([]float64))

		if i%1000 == 0 {
			fmt.Printf(
				"Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
				i,
				nn.Cost,
				accuracyValue)
		}

		//if 0.01 > cost.Value().Data().(float64) {
		//	break
		//}

	}

	nn.Save()

	fmt.Println("\n\nOutput after Training: \n", nn.PredVal)

	//nn.GetGraph().UnbindAllNonInputs()
	//x = gorgonia.NewMatrix(nn.GetGraph(),
	//	tensor.Float64,
	//	gorgonia.WithName("X"),
	//	gorgonia.WithShape(1, inputM.Ncol()),
	//)
	//nn.Fwd(x)
	fmt.Println("test------")
	tapeMachine.Reset()
	for i := 0; i < inputM.Nrow(); i++ {
		numRow := i
		batchSize := 1
		rowInput := inputM.Subset(makeRange(numRow, numRow+batchSize-1))
		rowOutput := outputM.Subset(makeRange(numRow, numRow+batchSize-1))
		xT := tensor.FromMat64(mat.DenseCopyOf(&matrix{rowInput}))
		yT := tensor.FromMat64(mat.DenseCopyOf(&matrix{rowOutput}))
		tapeMachine.Let(x, xT)
		tapeMachine.Let(y, yT)

		if err = tapeMachine.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", numRow, err)
		}

		tapeMachine.Reset()

		fmt.Println(rowInput.Records(), nn.PredVal.Data().([]float64)[i], rowOutput.Records())
	}

}

func makeRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

type DatasetIn struct {
	CouponId         float64
	AgeRange         float64
	MaritalStatus    float64
	Rented           float64
	FamilySize       float64
	NoOfChildren     float64
	IncomeBracket    float64
	RedemptionStatus float64
}

func getInputData() (*matrix, *matrix) {
	cfg := mysql.Config{
		User:                 "user",
		Passwd:               "password",
		Net:                  "tcp",
		Addr:                 "localhost:3306",
		DBName:               "db",
		AllowNativePasswords: true,
	}

	db, err := sql.Open("mysql", cfg.FormatDSN())
	if err != nil {
		log.Fatal(err)
	}

	rows, err := db.Query("SELECT coupon_id, age_range, marital_status, rented, family_size, no_of_children, income_bracket, redemption_status FROM train, customer_demographics WHERE train.customer_id=customer_demographics.customer_id")
	if err != nil {
		panic("albumsByArtist %q: %v")
	}
	defer rows.Close()

	var datasetIn []DatasetIn
	// Loop through rows, using Scan to assign column data to struct fields.
	for rows.Next() {
		var dataIn DatasetIn
		if err := rows.Scan(&dataIn.CouponId, &dataIn.AgeRange, &dataIn.MaritalStatus, &dataIn.Rented, &dataIn.FamilySize, &dataIn.NoOfChildren, &dataIn.IncomeBracket, &dataIn.RedemptionStatus); err != nil {
			panic(err)
		}
		datasetIn = append(datasetIn, dataIn)
	}
	if err := rows.Err(); err != nil {
		panic(err)
	}

	df := dataframe.LoadStructs(datasetIn)

	inputDf := df.Drop("RedemptionStatus")
	outputDf := df.Select("RedemptionStatus")

	return &matrix{inputDf}, &matrix{outputDf}
}

func getInputDataFromFile() (*matrix, *matrix) {
	fileName := "dataset/car_data.csv"
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	df := dataframe.ReadCSV(f)

	inputDf := df.Drop("Purchased")

	bias := make([]float64, inputDf.Nrow())
	for i := 0; i < inputDf.Nrow(); i++ {
		bias[i] = 1
	}
	inputDf = inputDf.Mutate(
		series.New(bias, series.Float, "Bias"),
	)
	outputDf := df.Select("Purchased")

	return &matrix{inputDf}, &matrix{outputDf}
}

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}
