FROM golang:1.21.1-alpine

WORKDIR /mlp

COPY go.mod .
COPY go.sum .
RUN go mod download

COPY . .
WORKDIR /mlp/cmd
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -installsuffix cgo -o ./cmd