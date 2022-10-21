package random

import (
	"math"
	"math/rand"
)

func Float64(min, max float64) float64 {
	upper := math.Ceil(min)
	lower := math.Floor(max)
	return rand.Float64()*(upper-lower) + lower
}
