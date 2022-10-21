package main

import (
	"fmt"
	"math/rand"
	. "micrograd"
	"time"
)

type Tensor interface {
	//Backward()
	Call()
}

func main() {
	rand.Seed(time.Now().Unix())

	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	m := NewSequential(
		NewLinear(3, 4),
		NewLinear(4, 4),
		NewLinear(4, 1),
	)

	var last []*Value
	for k := 1; k < 10000; k++ {
		var ypred []*Value

		for i := range xs {
			ypred = append(ypred, m.Call(xs[i]))
		}

		loss := NewScalar(0)
		for i, yout := range ypred {
			loss = loss.Add(yout.Sub(NewScalar(ys[i])).Pow(2))
		}

		for _, p := range m.Parameters() {
			p.Grad = 0.0
		}

		loss.Backward()

		for _, p := range m.Parameters() {
			p.Data += -0.1 * p.Grad
		}

		fmt.Printf("k: %d loss:%f\n", k, loss.Data)
		last = ypred
	}

	fmt.Println("Final predictions:")
	for _, pred := range last {
		pred.PrintValue()
	}
}
