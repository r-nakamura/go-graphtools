package graphtools

import (
	"reflect"
	"sort"
	"testing"
)

func sortVertices(V *[]int) {
	sort.Ints(*V)
}

func sortEdges(E *[][2]int) {
	sort.Slice(*E, func(i, j int) bool {
		if (*E)[i][0] != (*E)[j][0] {
			return (*E)[i][0] < (*E)[j][0]
		} else {
			return (*E)[i][1] < (*E)[j][1]
		}
	})
}

func TestGraphAttribute(t *testing.T) {
	var g Graph
	g.New()
	val := 1
	g.SetGraphAttribute("key", val)

	got := g.GraphAttribute("key")
	expected := val
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestVertices(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddVertex(1)
	g.AddVertex(2)
	g.AddVertex(3)

	got := g.Vertices()
	sortVertices(&got)
	expected := []int{1, 2, 3}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestHasVertex(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddVertex(1)
	if !g.HasVertex(1) {
		t.Error("HasVertex(1) = false")
	}
}

func TestPredecessors(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 3)
	g.AddEdge(2, 3)

	got := g.Predecessors(3)
	sortVertices(&got)
	expected := []int{1, 2}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestSuccessors(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.Successors(1)
	sortVertices(&got)
	expected := []int{2, 3}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestNeighbors(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.Successors(1)
	sortVertices(&got)
	expected := []int{2, 3}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestVertexAttribute(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddVertex(1)
	g.SetVertexAttribute(1, "degree", 0)

	got := g.VertexAttribute(1, "degree")
	expected := 0
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}

	if g.VertexAttribute(1, "betweenness") != nil {
		t.Error("VertexAttribute(1, key) = false")
	}
}

func TestVertexAttributes(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddVertex(1)
	attrs := map[string]interface{}{
		"degree":      1,
		"betweenness": 2,
	}
	g.SetVertexAttributes(1, attrs)

	got := g.VertexAttributes(1)
	expected := attrs
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestDeleteVertex(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddVertex(1)
	g.DeleteVertex(1)
	if g.HasVertex(1) {
		t.Error("DeleteVertex(1) = false")
	}
}

func TestEdges(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.Edges()
	sortEdges(&got)
	expected := [][2]int{{1, 2}, {1, 2}, {1, 3}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestUniqueEdges(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.UniqueEdges()
	sortEdges(&got)
	expected := [][2]int{{1, 2}, {1, 3}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestHasEdge(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	if !g.HasEdge(1, 2) {
		t.Error("HasEdge(1, 2) = false")
	}
}

func TestMultiEdgeIDs(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 2)

	got := g.MultiEdgeIDs(1, 2)
	expected := []int{0, 1}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestEdgeCount(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 2)

	got := g.EdgeCount(1, 2)
	expected := 2
	if got != expected {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestDeleteEdge(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.DeleteEdge(1, 2)
	if g.HasEdge(1, 2) {
		t.Error("DeleteEdge(1, 2) = false")
	}
}

func TestEdgesFrom(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.EdgesFrom(1)
	sortEdges(&got)
	expected := [][2]int{{1, 2}, {1, 3}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestEdgesTo(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(2, 1)
	g.AddEdge(3, 1)

	got := g.EdgesTo(1)
	sortEdges(&got)
	expected := [][2]int{{2, 1}, {3, 1}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestEdgesAt(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(3, 1)

	got := g.EdgesAt(1)
	sortEdges(&got)
	expected := [][2]int{{1, 2}, {3, 1}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestOutDegree(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(1, 3)

	got := g.OutDegree(1)
	expected := 2
	if got != expected {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestInDegree(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(2, 1)
	g.AddEdge(3, 1)

	got := g.InDegree(1)
	expected := 2
	if got != expected {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestDegree(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(3, 1)

	got := g.Degree(1)
	expected := 2
	if got != expected {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestEdgeAttributeByID(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.SetEdgeAttributeByID(1, 2, 0, "weight", 1)

	got := g.EdgeAttributeByID(1, 2, 0, "weight")
	expected := 1
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}

	if g.EdgeAttributeByID(1, 2, 0, "cost") != nil {
		t.Error("EdgeAttributeByID(1, 2, key) = false")
	}
}

func TestVertexAttributesByID(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	attrs := map[string]interface{}{
		"weight":   1,
		"capacity": 2,
	}
	g.SetEdgeAttributesByID(1, 2, 0, attrs)

	got := g.EdgeAttributesByID(1, 2, 0)
	expected := attrs
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestDijkstra(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(1, 3)
	g.AddEdge(3, 4)

	dist, prev := g.Dijkstra(1)
	expected1 := map[int]float64{
		1: 0,
		2: 1,
		3: 1,
		4: 2,
	}
	if !reflect.DeepEqual(dist, expected1) {
		t.Errorf("got: %v, expected: %v", dist, expected1)
	}

	expected2 := map[int][]int{
		1: []int{},
		2: []int{1},
		3: []int{1},
		4: []int{3},
	}
	if !reflect.DeepEqual(prev, expected2) {
		t.Errorf("got: %v, expected: %v", dist, expected2)
	}
}

func TestShortestPaths(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(1, 3)

	got := g.ShortestPaths(1, 3)
	expected := [][]int{{1, 3}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestFloydWarshall(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(1, 3)
	g.AddEdge(3, 4)

	g.FloydWarshall()

	got := g.T
	expected := map[int]map[int]float64{
		1: map[int]float64{
			2: 1, 3: 1, 4: 2,
		},
		2: map[int]float64{
			3: 1, 4: 2,
		},
		3: map[int]float64{
			4: 1,
		},
		4: map[int]float64{
		},
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestIsReachable(t *testing.T) {
	var g DirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(3, 4)
	if g.IsReachable(1, 4) != true {
		t.Error("IsReachable(1, 4) = false")
	}

	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(3, 4)
	if g.IsReachable(1, 4) != false {
		t.Error("IsReachable(1, 4) = true")
	}
}

func TestExplore(t *testing.T) {
	var g UndirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(3, 4)
	g.AddEdge(4, 1)

	g.AddEdge(5, 6)
	g.AddEdge(6, 7)
	g.AddEdge(7, 5)

	got := g.Explore(1)
	sortVertices(&got)
	expected := []int{1, 2, 3, 4}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestIsConnected(t *testing.T) {
	var g UndirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(3, 4)
	g.AddEdge(4, 1)
	if g.IsConnected() == false {
		t.Error("IsConnected() = true")
	}

	g.AddEdge(5, 6)
	if g.IsConnected() == true {
		t.Error("IsConnected() = false")
	}
}

func TestMaximalComponent(t *testing.T) {
	var g UndirectedGraph
	g.New()
	g.AddEdge(1, 2)
	g.AddEdge(2, 3)
	g.AddEdge(3, 4)
	g.AddEdge(4, 1)

	g.AddEdge(5, 6)
	g.AddEdge(6, 7)
	g.AddEdge(7, 5)

	got := g.MaximalComponent()
	sortVertices(&got)
	expected := []int{1, 2, 3, 4}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}

func TestCompleteGraph(t *testing.T) {
	var g UndirectedGraph
	g.New()
	g.AddVertices([]int{1, 2, 3, 4})
	g.CompleteGraph()

	got := g.Edges()
	sortEdges(&got)
	expected := [][2]int{
		{1, 2}, {1, 3}, {1, 4},
		{2, 3}, {2, 4},
		{3, 4},
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got: %v, expected: %v", got, expected)
	}
}
