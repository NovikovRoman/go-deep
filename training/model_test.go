package training

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_SplitSize(t *testing.T) {
	e := make(Examples, 10)
	batches := e.SplitSize(15)
	assert.Len(t, batches, 1)
	assert.Equal(t, 10, len(batches[0]))

	batches = e.SplitSize(2)
	assert.Len(t, batches, 5)
	for _, batch := range batches {
		assert.Equal(t, 2, len(batch))
	}

	batches = e.SplitSize(3)
	assert.Len(t, batches, 4)
	for i, batch := range batches {
		if i == 3 {
			assert.Equal(t, 1, len(batch))
		} else {
			assert.Equal(t, 3, len(batch))
		}
	}
}

func Test_SplitN(t *testing.T) {
	e := make(Examples, 10)

	partitions := e.SplitN(3)
	assert.Len(t, partitions, 3)
	assert.Len(t, partitions[0], 4)
	assert.Len(t, partitions[1], 3)
	assert.Len(t, partitions[2], 3)
}

func Test_Split(t *testing.T) {
	rand.New(rand.NewSource(0))

	e := make(Examples, 100)
	a, b := e.Split(0.5)

	assert.InEpsilon(t, len(a), 50, 0.1)
	assert.InEpsilon(t, len(b), 50, 0.1)
}

func TestExamples_SplitSize(t *testing.T) {
	type args struct {
		size int
	}
	tests := []struct {
		name    string
		e       Examples
		args    args
		wantRes []Examples
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRes := tt.e.SplitSize(tt.args.size); !reflect.DeepEqual(gotRes, tt.wantRes) {
				t.Errorf("Examples.SplitSize() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
