package main

import (
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"os"
	"slices"
	"strconv"
)

//var err error

func main() {
	file, err := os.Open("car_data_orig.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	df := dataframe.ReadCSV(file)

	df.Describe()

	dfFited := fitBrandType(df)
	dfFited = fitCols(dfFited)

	updatedFile, err := os.Create("car_data.csv")
	err = dfFited.WriteCSV(updatedFile)
	if err != nil {
		panic(err)
	}
}

func fitCols(df dataframe.DataFrame) dataframe.DataFrame {
	changeF1 := func(s series.Series) series.Series {
		if s.Name != "AnnualSalary" {
			return s
		}

		records := s.Records()
		newRecords := make([]float64, len(records))
		for i, v := range records {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				panic(err)
			}

			newRecords[i] = f / 100000.0
		}

		return series.Floats(newRecords)
	}

	column := df.Capply(changeF1)

	changeF2 := func(s series.Series) series.Series {
		if s.Name != "Age" {
			return s
		}

		records := s.Records()
		newRecords := make([]float64, len(records))
		for i, v := range records {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				panic(err)
			}

			newRecords[i] = f / 100.0
		}

		return series.Floats(newRecords)
	}

	column2 := column.Capply(changeF2)

	changeF3 := func(s series.Series) series.Series {
		if s.Name != "UserId" {
			return s
		}

		records := s.Records()
		newRecords := make([]float64, len(records))
		for i, v := range records {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				panic(err)
			}

			newRecords[i] = f / 1000.0
		}

		return series.Floats(newRecords)
	}

	column3 := column2.Capply(changeF3)

	return column3
}

func fitSalary(df dataframe.DataFrame) dataframe.DataFrame {
	dense1 := mat.DenseCopyOf(&matrix{df.Select("AnnualSalary")})
	dense2 := mat.DenseCopyOf(&matrix{df.Select("Age")})
	pt1 := preprocessing.NewPowerTransformer()
	dense1, _ = pt1.FitTransform(dense1, dense1)

	pt2 := preprocessing.NewPowerTransformer()
	dense2, _ = pt2.FitTransform(dense2, dense2)

	//var vv []float64
	//for i := 0; i < v.Len(); i++ {
	//	vv = append(vv, v.AtVec(i))
	//}

	vdf := dataframe.LoadMatrix(dense1.ColView(0))
	changeF1 := func(s series.Series) series.Series {
		if s.Name != "AnnualSalary" {
			return s
		}

		return series.Floats(vdf.Col("X0").Float())
	}

	column := df.Capply(changeF1)

	vdf2 := dataframe.LoadMatrix(dense2.ColView(0))
	changeF2 := func(s series.Series) series.Series {
		if s.Name != "Age" {
			return s
		}

		return series.Floats(vdf2.Col("X0").Float())
	}

	column2 := column.Capply(changeF2)

	return column2
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

func fitBrandType(df dataframe.DataFrame) dataframe.DataFrame {
	dfAge := df.GroupBy("brand_type")
	var groups []string

	for i := range dfAge.GetGroups() {
		groups = append(groups, i)
	}

	//groups := []string{"1", "2", "3", "4", "5+"}

	changeF := func(s series.Series) series.Series {
		if s.Name != "brand_type" {
			return s
		}

		records := s.Records()
		newRecords := make([]float64, len(records))
		for i, v := range records {
			newRecords[i] = float64(slices.Index(groups, v) + 1)
		}
		return series.Floats(newRecords)
	}

	column := df.Capply(changeF)
	return column
}
