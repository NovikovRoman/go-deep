package training

import (
	"math"
	"math/rand"
)

// Example is an input-target pair
type Example struct {
	Input    []float64
	Response []float64
}

// Examples is a set of input-output pairs
type Examples []Example

// Shuffle shuffles slice in-place
func (e Examples) Shuffle() {
	for i := range e {
		j := rand.Intn(i + 1)
		e[i], e[j] = e[j], e[i]
	}
}

// Split assigns each element to two new slices
// according to probability p
func (e Examples) Split(p float64) (first, second Examples) {
	for i := 0; i < len(e); i++ {
		if p > rand.Float64() {
			first = append(first, e[i])
		} else {
			second = append(second, e[i])
		}
	}
	return
}

// SplitSize splits slice into parts of size size
func (e Examples) SplitSize(size int) (res []Examples) {
	//s := len(e) / size
	s := int(math.Ceil(float64(len(e)) / float64(size)))
	if s == 0 {
		s++
	}

	res = make([]Examples, s)
	for i := 0; i < len(e); i += size {
		res[i/size] = e[i:min(i+size, len(e))]
	}
	return
}

// SplitN splits slice into n parts
func (e Examples) SplitN(n int) []Examples {
	res := make([]Examples, n)
	for i, el := range e {
		res[i%n] = append(res[i%n], el)
	}
	return res
}
