package micrograd

import (
	"micrograd/random"
)

type Neuron struct {
	w []*Value
	b *Value
}

func NewNeuron(in int) *Neuron {
	n := &Neuron{}
	n.w = make([]*Value, in)
	n.b = &Value{Data: random.Float64(-1, 1), label: "b"}
	for i := 0; i < in; i++ {
		n.w[i] = &Value{Data: random.Float64(-1, 1)}
	}
	return n
}

func (n Neuron) Call(x []*Value) *Value {
	r := n.b
	for i, _ := range n.w {
		r = r.Add(n.w[i].Mult(x[i]))
	}
	out := r.Tanh()
	return out
}

func (n Neuron) Parameters() []*Value {
	return append(n.w, n.b)
}

type Layer interface {
	Call(vals []*Value) []*Value
	Parameters() []*Value
}

type Linear struct {
	neurons []*Neuron
}

func NewLinear(in, out int) *Linear {
	l := &Linear{}
	l.neurons = make([]*Neuron, out)
	for i := 0; i < out; i++ {
		l.neurons[i] = NewNeuron(in)
	}
	return l
}

func (l Linear) Call(x []*Value) []*Value {
	var outs []*Value
	for _, n := range l.neurons {
		outs = append(outs, n.Call(x))
	}
	return outs
}

func (l Linear) Parameters() []*Value {
	var params []*Value

	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}

	return params
}

type Sequential struct {
	layers []Layer
}

type MLP struct {
	layers []Layer
}

func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{layers: layers}
}

func (s Sequential) Call(x []float64) *Value {
	var vals []*Value
	for _, el := range x {
		vals = append(vals, NewScalar(el))
	}

	for _, layer := range s.layers {
		vals = layer.Call(vals)
	}

	return vals[0]
}

func (s Sequential) Parameters() []*Value {
	var params []*Value

	for _, layer := range s.layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}

func NewMLP(in int, outs []int) *MLP {

	sz := append([]int{in}, outs...)
	var layers []Layer
	for i, _ := range outs {
		layers = append(layers, NewLinear(sz[i], sz[i+1]))
	}
	m := &MLP{layers: layers}
	return m
}

func (m MLP) Call(x []float64) *Value {

	var vals []*Value
	for _, el := range x {
		vals = append(vals, NewScalar(el))
	}

	for _, layer := range m.layers {
		vals = layer.Call(vals)
	}

	return vals[0]
}

func (m MLP) Parameters() []*Value {
	var params []*Value

	for _, layer := range m.layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}
