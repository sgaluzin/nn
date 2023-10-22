# Neural network for predict car purchase
Multi layer perceptron for with one hidden layer for predict purchase a car depends on gender, age and salary

## How to execute already trained model
```bash
docker buildx build -t car_predict .
docker run -it --name car_predict car_predict
./cmd
```

## How to retrain model
```bash
docker buildx build -t car_predict .
docker run -it --name car_predict car_predict
cd ..
go run .
```

