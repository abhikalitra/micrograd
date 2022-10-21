package micrograd

import (
	"fmt"
	"math"
)

type Value struct {
	Data         float64
	Grad         float64
	label        string
	op           string
	prev         []*Value
	backward     func()
	visited      bool
	requiresGrad bool
}

func NewScalar(data float64) *Value {
	return &Value{Data: data}
}

func NewValue(data float64, op string, children ...*Value) *Value {
	return &Value{Data: data, op: op, prev: children}
}

func (a *Value) Add(b *Value) *Value {
	out := NewValue(a.Data+b.Data, "+", a, b)

	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}
	return out
}

func (a *Value) Sub(b *Value) *Value {
	out := a.Add(b.Neg())
	out.op = "-"
	out.prev = []*Value{a, b}
	return out
}

func (a *Value) Mult(b *Value) *Value {
	out := NewValue(a.Data*b.Data, "*", a, b)
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}

func (a *Value) Neg() *Value {
	v2 := NewScalar(-1)
	out := NewValue(a.Data*v2.Data, "-", a, v2)
	return out
}

func (a *Value) Pow(other float64) *Value {
	out := NewValue(math.Pow(a.Data, other), "pow", a)
	out.backward = func() {
		a.Grad += (other * math.Pow(a.Data, other-1)) * out.Grad
	}
	return out
}

func (a *Value) Div(v2 *Value) *Value {
	out := NewValue(a.Data*v2.Pow(-1).Data, "/", a, v2)
	return out
}

func (a *Value) MultS(b float64) *Value {
	out := NewValue(a.Data*b, "*", a)
	out.backward = func() {
		a.Grad += b * out.Grad
	}
	return out
}

func (a *Value) Relu() *Value {
	val := 0.0
	grad := 0.0
	if a.Data > 0 {
		val = a.Data
		grad = 1.0
	}
	out := NewValue(val, "relu", a)
	out.backward = func() {
		a.Grad += grad * out.Grad
	}
	return out
}

func (a *Value) Tanh() *Value {
	t := (math.Exp(2*a.Data) - 1) / (math.Exp(2*a.Data) + 1)
	out := NewValue(t, "tanh", a)
	out.backward = func() {
		a.Grad += (1 - (t * t)) * out.Grad
	}
	return out
}

func (a *Value) Exp() *Value {
	t := math.Exp(a.Data)
	out := NewValue(t, "exp", a)
	out.backward = func() {
		a.Grad += t * out.Grad
	}
	return out
}

func (a *Value) Backward() {

	a.Grad = 1
	var topo []*Value
	var visited []*Value
	var f func(v *Value)

	f = func(v *Value) {
		if !v.visited {
			v.visited = true
			visited = append(visited, v)
			for _, child := range v.prev {
				f(child)
			}
			topo = append(topo, v)
		}
	}
	f(a)

	for i := len(topo) - 1; i > 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

func (a *Value) Print(levels ...int) {
	level := 0
	pre := ""
	if len(levels) > 0 {
		level = levels[0]
		for i := 0; i < level; i++ {
			pre += "   "
		}
	}
	fmt.Printf("%s----------------------------------\n", pre)
	fmt.Printf("%s%s | %s | %.4f | %.4f \n", pre, a.label, a.op, a.Data, a.Grad)
	fmt.Printf("%s---------------------------------\n", pre)
	for _, child := range a.prev {
		child.Print(level + 1)
	}
}

func (v Value) PrintValue() {
	fmt.Printf("( data=%f, grad=%f, op=%s) ", v.Data, v.Grad, v.op)
	//fmt.Println("prev=", v.prev, ")")
}
