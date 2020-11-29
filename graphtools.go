package graphtools

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	version = "0.1.0"
)

const (
	MAX_RETRIES = 100
	INFINITY    = math.MaxFloat64
)

type Graph struct {
	V        map[int]Vertex
	EI, EO   map[int]map[int]Edges
	T        map[int]map[int]float64
	P        map[int]map[int][]int
	Cb       map[int]float64
	directed bool
	attr     Attribute
}

type Vertex struct {
	v    int
	attr Attribute
}

type Edge struct {
	u, v int
	id   int
	attr Attribute
}
type Edges []Edge

type Attribute map[string]interface{}

func (g *Graph) New() {
	g.V = make(map[int]Vertex)
	g.EI = make(map[int]map[int]Edges)
	g.EO = make(map[int]map[int]Edges)
	g.T = make(map[int]map[int]float64)
	g.P = make(map[int]map[int][]int)
	g.Cb = make(map[int]float64)
}

type DirectedGraph struct {
	Graph
}

func (g *DirectedGraph) New() {
	g.Graph.New()
	g.directed = true
}

type UndirectedGraph struct {
	Graph
}

func (g *UndirectedGraph) New() {
	g.Graph.New()
	g.directed = false
}

func (g *Graph) Directed() bool { return g.directed }

func (g *Graph) Undirected() bool { return !g.directed }

func (g *Graph) ExpectDirected() {
	if !g.Directed() {
		die("directed graph expected")
	}
}

func (g *Graph) ExpectUndirected() {
	if !g.Undirected() {
		die("undirected graph expected")
	}
}

func (g *Graph) SetGraphAttribute(key string, val interface{}) {
	g.attr[key] = val
}

func (g *Graph) GraphAttribute(key string) interface{} {
	if val, ok := g.attr[key]; ok {
		return val
	}
	return nil
}

func (g *Graph) AverageDegree() float64 {
	var d int
	V := g.Vertices()
	for _, v := range V {
		d += g.Degree(v)
	}
	if len(V) > 0 {
		return float64(d) / float64(len(V))
	} else {
		return 0
	}
}

func (g *Graph) Vertices() []int {
	var vertices []int
	for v := range g.V {
		vertices = append(vertices, v)
	}
	return vertices
}

func (g *Graph) HasVertex(v int) bool {
	if _, ok := g.V[v]; ok {
		return true
	} else {
		return false
	}
}

func (g *Graph) AddVertex(v int) {
	if !g.HasVertex(v) {
		g.V[v] = Vertex{v: v, attr: make(map[string]interface{})}
	}
}

func (g *Graph) AddVertices(vertices []int) {
	for _, v := range vertices {
		g.AddVertex(v)
	}
}

func (g *Graph) Predecessors(v int) []int {
	var found []int
	for u := range g.EI[v] {
		found = append(found, u)
	}
	return found
}

func (g *Graph) Successors(u int) []int {
	var found []int
	for v := range g.EO[u] {
		found = append(found, v)
	}
	return found
}

func (g *Graph) Neighbors(v int) []int {
	found := make(map[int]bool)
	for _, u := range g.Predecessors(v) {
		found[u] = true
	}
	for _, u := range g.Successors(v) {
		found[u] = true
	}
	var neighbors []int
	for u := range found {
		neighbors = append(neighbors, u)
	}
	return neighbors
}

func (g *Graph) SetVertexAttribute(v int, key string, val interface{}) {
	if !g.HasVertex(v) {
		die(fmt.Sprintf("SetVertexAttribute: no vertex %v", v))
	}
	g.V[v].attr[key] = val
}

func (g *Graph) VertexAttribute(v int, key string) interface{} {
	if val, ok := g.V[v].attr[key]; ok {
		return val
	}
	return nil
}

func (g *Graph) SetVertexAttributes(v int, attrs map[string]interface{}) {
	for key, val := range attrs {
		g.SetVertexAttribute(v, key, val)
	}
}

func (g *Graph) VertexAttributes(v int) map[string]interface{} {
	m := make(map[string]interface{})
	for key, val := range g.V[v].attr {
		m[key] = val
	}
	return m
}

func (g *Graph) SetVertexWeight(v int, val interface{}) {
	g.SetVertexAttribute(v, "weight", val)
}

func (g *Graph) VertexWeight(v int) interface{} {
	return g.VertexAttribute(v, "weight")
}

func (g *Graph) DeleteVertex(v int) {
	if !g.HasVertex(v) {
		return
	}
	for u := range g.Neighbors(v) {
		delete(g.EO[u], v)
		delete(g.EI[u], v)
	}
	delete(g.V, v)
	delete(g.EO, v)
	delete(g.EI, v)
}

func (g *Graph) DeleteVertices(vertices []int) {
	for v := range vertices {
		g.DeleteVertex(v)
	}
}

func (g *Graph) RandomVertex() int {
	V := g.Vertices()
	return V[rand.Intn(len(V))]
}

func (g *Graph) Edges() [][2]int {
	var found [][2]int
	for u := range g.EO {
		for v := range g.EO[u] {
			for _ = range g.EO[u][v] {
				e := [2]int{u, v}
				found = append(found, e)
			}
		}
	}
	return found
}

func (g *Graph) UniqueEdges() [][2]int {
	var found [][2]int
	for u := range g.EO {
		for v := range g.EO[u] {
			e := [2]int{u, v}
			found = append(found, e)
		}
	}
	return found
}

func (g *Graph) HasEdge(u, v int) bool {
	if g.Undirected() && u > v {
		u, v = v, u
	}
	if _, ok := g.EO[u]; !ok {
		return false
	}
	if _, ok := g.EO[u][v]; !ok {
		return false
	}
	return true
}

func (g *Graph) MultiEdgeIDs(u, v int) []int {
	var ids []int
	if g.Undirected() && u > v {
		u, v = v, u
	}
	if !g.HasEdge(u, v) {
		return []int{}
	}
	for _, e := range g.EO[u][v] {
		ids = append(ids, e.id)
	}
	return ids
}

func (g *Graph) EdgeCount(u, v int) int {
	ids := g.MultiEdgeIDs(u, v)
	if len(ids) > 0 {
		return len(ids)
	} else {
		return 0
	}
}

func (g *Graph) AddEdge(u, v int) {
	if g.Undirected() && u > v {
		u, v = v, u
	}

	g.AddVertices([]int{u, v})

	id := g.EdgeCount(u, v)
	e := Edge{u: u, v: v, id: id, attr: make(map[string]interface{})}
	if _, ok := g.EO[u]; !ok {
		g.EO[u] = make(map[int]Edges)
	}
	g.EO[u][v] = append(g.EO[u][v], e)
	if _, ok := g.EI[v]; !ok {
		g.EI[v] = make(map[int]Edges)
	}
	g.EI[v][u] = append(g.EI[v][u], e)
}

func (g *Graph) DeleteEdge(u, v int) {
	if g.Undirected() && u > v {
		u, v = v, u
	}
	if !g.HasEdge(u, v) {
		return
	}
	if g.EdgeCount(u, v) > 1 {
		g.EO[u][v] = g.EO[u][v][:len(g.EO[u][v])-1]
		g.EI[v][u] = g.EI[v][u][:len(g.EO[u][v])-1]
	} else {
		delete(g.EO[u], v)
		delete(g.EI[v], u)
	}
}

func (g *Graph) EdgesFrom(u int) [][2]int {
	var found [][2]int
	for _, v := range g.Successors(u) {
		for range g.EO[u][v] {
			e := [2]int{u, v}
			found = append(found, e)
		}
	}
	return found
}

func (g *Graph) EdgesTo(v int) [][2]int {
	var found [][2]int
	for _, u := range g.Predecessors(v) {
		for range g.EI[v][u] {
			e := [2]int{u, v}
			found = append(found, e)
		}
	}
	return found
}

func (g *Graph) EdgesAt(v int) [][2]int {
	var found [][2]int
	for _, e := range g.EdgesFrom(v) {
		found = append(found, e)
	}
	for _, e := range g.EdgesTo(v) {
		found = append(found, e)
	}
	return found
}

func (g *Graph) OutDegree(u int) int {
	return len(g.EdgesFrom(u))
}

func (g *Graph) InDegree(v int) int {
	return len(g.EdgesTo(v))
}

func (g *Graph) Degree(v int) int {
	return g.InDegree(v) + g.OutDegree(v)
}

func (g *Graph) RandomEdge() [2]int {
	E := g.Edges()
	return E[rand.Intn(len(E))]
}

func (g *Graph) SetEdgeAttributeByID(u, v, n int, key string, val interface{}) {
	if g.Undirected() && u > v {
		u, v = v, u
	}
	if !g.HasEdge(u, v) {
		die(fmt.Sprintf("SetEdgeAttribute: edge (%v, %v) not found", u, v))
	}
	for _, e := range g.EO[u][v] {
		if e.id == n {
			e.attr[key] = val
		}
	}
}

func (g *Graph) EdgeAttributeByID(u, v, n int, key string) interface{} {
	if g.Undirected() && u > v {
		u, v = v, u
	}
	for _, e := range g.EO[u][v] {
		if e.id == n {
			if val, ok := e.attr[key]; ok {
				return val
			}
		}
	}
	return nil
}

func (g *Graph) SetEdgeAttributesByID(u, v, n int, attrs map[string]interface{}) {
	for key, val := range attrs {
		g.SetEdgeAttributeByID(u, v, n, key, val)
	}
}

func (g *Graph) EdgeAttributesByID(u, v, n int) map[string]interface{} {
	attrs := make(map[string]interface{})
	for _, e := range g.EO[u][v] {
		if e.id == n {
			for key, val := range e.attr {
				attrs[key] = val
			}
		}
	}
	return attrs
}

func (g *Graph) SetEdgeWeightByID(u, v, n int, w interface{}) {
	g.SetEdgeAttributeByID(u, v, n, "weight", w)
}

func (g *Graph) EdgeWeightByID(u, v, n int) interface{} {
	return g.EdgeAttributeByID(u, v, n, "weight")
}

func (g *Graph) SetEdgeWeight(u, v int, w interface{}) {
	g.SetEdgeAttributeByID(u, v, 0, "weight", w)
}

func (g *Graph) EdgeWeight(u, v int) interface{} {
	return g.EdgeAttributeByID(u, v, 0, "weight")
}

func (g *Graph) Dijkstra(s int) (map[int]float64, map[int][]int) {
	g.ExpectDirected()

	dist := make(map[int]float64)
	prev := make(map[int][]int)
	for _, v := range g.Vertices() {
		prev[v] = []int{}
	}
	dist[s] = 0

	Q := g.Vertices()
	for len(Q) > 0 {
		sortByDist(&Q, dist)
		u := Q[0]
		Q = Q[1:]
		if _, ok := dist[u]; !ok {
			break
		}

		for _, v := range g.Successors(u) {
			var w, d float64
			if g.EdgeWeightByID(u, v, 0) != nil {
				w = g.EdgeWeightByID(u, v, 0).(float64)
			} else {
				w = 1
			}
			if _, ok := dist[u]; ok {
				d = dist[u]
			} else {
				d = INFINITY
			}

			if _, ok := dist[v]; !ok || dist[v] > d+w {
				dist[v] = dist[u] + w
				prev[v] = []int{u}
			} else if dist[v] == d+w {
				prev[v] = append(prev[v], u)
			}
		}
	}
	g.T[s], g.P[s] = dist, prev
	return dist, prev
}

func sortByDist(V *[]int, dist map[int]float64) {
	type T struct {
		v int
		d float64
	}
	U := make([]T, len(*V))
	for i, v := range *V {
		if _, ok := dist[v]; ok {
			U[i] = T{v: v, d: dist[v]}
		} else {
			U[i] = T{v: v, d: INFINITY}
		}
	}
	sort.Slice(U, func(i, j int) bool { return U[i].d < U[j].d })
	for i, _ := range *V {
		(*V)[i] = U[i].v
	}
}

func (g *Graph) ShortestPaths(s, t int) [][]int {
	g.ExpectDirected()
	if _, ok := g.P[s]; !ok {
		g.Dijkstra(s)
	}
	var P [][]int
	g.findPath(s, t, &P, nil)
	return P
}

func (g *Graph) findPath(s, t int, P *[][]int, p []int) {
	pp := p
	pp = append(pp, t)
	for _, prev := range g.P[s][t] {
		if prev == s {
			pp = append(pp, s)
			for i, j := 0, len(pp)-1; i < j; i, j = i+1, j-1 {
				pp[i], pp[j] = pp[j], pp[i]
			}
			*P = append(*P, pp)
			return
		} else {
			g.findPath(s, prev, P, pp)
		}
	}
}

func (g *Graph) DijkstraAllPairs() {
	for _, v := range g.Vertices() {
		g.Dijkstra(v)
	}
}

func (g *Graph) FloydWarshall() {
}

func (g *Graph) IsReachable(u, v int) bool {
	if _, ok := g.T[u]; !ok {
		g.Dijkstra(u)
	}
	if _, ok := g.T[u][v]; ok {
		return true
	} else {
		return false
	}
}

func (g *Graph) Explore(s int) []int {
	explored := make(map[int]bool)
	var needVisit = []int{}
	needVisit = append(needVisit, s)

	for len(needVisit) > 0 {
		u := needVisit[len(needVisit)-1]
		needVisit = needVisit[:len(needVisit)-1]
		explored[u] = true
		for _, v := range g.Neighbors(u) {
			if !explored[v] {
				needVisit = append(needVisit, v)
			}
		}
	}

	V := []int{}
	for v, _ := range explored {
		V = append(V, v)
	}
	return V
}

func (g *Graph) IsConnected() bool {
	v := g.RandomVertex()
	explored := g.Explore(v)
	return len(explored) == len(g.Vertices())
}

func (g *Graph) Components() [][]int {
	var components [][]int
	unvisited := make(map[int]bool)
	for _, v := range g.Vertices() {
		unvisited[v] = true
	}
	keys := func(m map[int]bool) []int {
		var a []int
		for v, _ := range m {
			a = append(a, v)
		}
		return a
	}

	for {
		V := keys(unvisited)
		if len(V) == 0 {
			break
		}
		s := V[rand.Intn(len(V))]
		explored := g.Explore(s)
		components = append(components, explored)
		for _, u := range explored {
			delete(unvisited, u)
		}
	}

	return components
}

func (g *Graph) MaximalComponent() []int {
	co := g.Components()
	sort.Slice(co, func(i, j int) bool { return len(co[i]) > len(co[j]) })
	return co[0]
}

func (g *Graph) Betweenness(v int) float64 {
	if _, ok := g.Cb[v]; !ok {
		g.updateBetweenness()
	}
	return g.Cb[v]
}

func (g *Graph) updateBetweenness() {
	g.ExpectUndirected()

	for _, v := range g.Vertices() {
		g.Cb[v] = 0
	}

	for _, s := range g.Vertices() {
		var S []int // empty stack
		P := make(map[int][]int)
		sigma := make(map[int]float64)
		sigma[s] = 1
		d := make(map[int]int)
		for _, t := range g.Vertices() {
			d[t] = -1
		}
		d[s] = 1
		var Q []int // empty queue
		Q = append(Q, s)

		for len(Q) > 0 {
			v := Q[0]
			Q = Q[1:]
			S = append(S, v)
			for _, w := range g.Neighbors(v) {
				if d[w] < 0 {
					Q = append(Q, w)
					d[w] = d[v] + 1
				}
				if d[w] == d[v]+1 {
					sigma[w] += sigma[v]
					P[w] = append(P[w], v)
				}
			}
		}

		delta := make(map[int]float64)
		for len(S) > 0 {
			w := S[len(S)-1]
			S = S[:len(S)-1]
			for _, v := range P[w] {
				delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
			}
			if w != s {
				g.Cb[w] += delta[w]
			}
		}
	}
}

func (g *Graph) CompleteGraph() {
	for _, u := range g.Vertices() {
		for _, v := range g.Vertices() {
			if u >= v {
				continue
			}
			if !g.HasEdge(u, v) {
				g.AddEdge(u, v)
			}
		}
	}
}

func (g *Graph) CreateRandomGraph(N, E int) {
	if E < N {
		die("CreateRandomGraph: too small number of edges")
	}

	for v := 1; v < N+1; v++ {
		g.AddVertex(v)
	}

	// add first (N - 1) edges for making sure connectivity
	for i := 1; i < N; i++ {
		u := i + 1
		v := rand.Intn(u-1) + 1
		if rand.Float64() >= 0.5 {
			g.AddEdge(u, v)
		} else {
			g.AddEdge(v, u)
		}
	}

	// randomly add remaining (E - (N - 1)) edges
	for i := 1; i < E-(N-1)+1; i++ {
		ntries := 1
		var u, v int
		for ntries < MAX_RETRIES {
			u = rand.Intn(N) + 1
			v = rand.Intn(N) + 1
			if u != v {
				break
			}
			if u != v && !g.HasEdge(u, v) {
				break
			}
		}
		g.AddEdge(u, v)
	}
}

func (g *Graph) CreateErdosRenyiGraph(N int, p float64) {
	g.ExpectUndirected()
	for v := 1; v < N+1; v++ {
		g.AddVertex(v)
	}
	for u := 1; u < N+1; u++ {
		for v := u + 1; v < N+1; v++ {
			if rand.Float64() < p {
				g.AddEdge(u, v)
			}
		}
	}
}

func (g *Graph) CreateRandomSparseGraph(N, E int, no_multiedge bool) {
	for v := 1; v < N+1; v++ {
		g.AddVertex(v)
	}

	for i := 0; i < E; i++ {
		ntries := 1
		var u, v int
		for ntries < MAX_RETRIES {
			u = rand.Intn(N) + 1
			v = rand.Intn(N) + 1
			if !no_multiedge && u != v {
				break
			}
			if !no_multiedge && u != v && !g.HasEdge(u, v) {
				break
			}
		}
		g.AddEdge(u, v)
	}
}

func (g *Graph) CreateBarabasiGraph(N, m0, m int) {
	g.ExpectUndirected()

	for v := 1; v < m0+1; v++ {
		g.AddVertex(v)
	}
	g.CompleteGraph()

	step := N - m0
	for s := 1; s < step+1; s++ {
		u := m0 + s
		g.AddVertex(u)

		edges := g.Edges()
		for i := 1; i < m+1; i++ {
			edge := edges[rand.Intn(len(edges))]
			v := edge[rand.Intn(2)]
			g.AddEdge(u, v)
		}
	}
}

func (g *Graph) CreateBarabasiRandomGraph(N, E, m0 int) {
	g.ExpectUndirected()

	for v := 1; v < m0+1; v++ {
		g.AddVertex(v)
	}
	g.CompleteGraph()

	E0 := m0 * (m0 - 1) / 2
	nedges := float64(E-E0) / float64(N-m0)

	for u := m0 + 1; u < N+1; u++ {
		g.AddVertex(u)

		E := g.Edges()
		for {
			e := E[rand.Intn(len(E))]
			v := e[rand.Intn(2)]
			g.AddEdge(u, v)

			if rand.Float64() <= 1/nedges {
				break
			}
		}
	}
}

func (g *Graph) CreateRingGraph(N, step int) {
	for v := 1; v < N+1; v++ {
		g.AddVertex(v)
	}

	for i := 0; i < N; i++ {
		u := i + 1
		v := ((i + step) % N) + 1
		g.AddEdge(u, v)
	}
}

func (g *Graph) CreateTreeGraph(N int) {
	g.AddVertex(1)
	for v := 2; v < N+1; v++ {
		u := rand.Intn(v-1) + 1
		g.AddEdge(u, v)
	}
}

func (g *Graph) CreateBtreeGraph(N int) {
	var depth float64
	nedges := 1
	finished := false
	for !finished {
		vleft := int(math.Pow(2, depth))
		for n := 1; n < int(math.Pow(2, depth))+1; n++ {
			v := vleft + (n - 1)
			parent := v / 2
			if parent == 0 {
				continue
			}
			g.AddEdge(v, parent)
			nedges++
			if nedges > N {
				finished = true
				break
			}
		}
		depth++
	}
}

func (g *Graph) CreateTreebaGraph(N int, alpha float64) {
	g.ExpectDirected()
	attract := make([]float64, N+1)

	g.AddVertex(1)
	attract[1] = alpha + float64(g.InDegree(1))

	for u := 2; u < N+1; u++ {
		var total float64
		for _, a := range attract {
			total += a
		}
		frac := rand.Float64() * total
		var sum float64
		for v := 1; v < N+1; v++ {
			sum += attract[v]
			if frac < sum {
				g.AddEdge(u, v)
				attract[u] = alpha + float64(g.InDegree(u))
				attract[v] = alpha + float64(g.InDegree(v))
				break
			}
		}
	}
}

func (g *Graph) CreateGeneralizedBarabasiGraph(N, m0, m int, gamma float64) {
	g.ExpectDirected()
	A := float64(m) * (gamma - 2)

	for v := 1; v < m0+1; v++ {
		g.AddVertex(v)
	}
	g.CompleteGraph()

	step := N - m0
	for s := 1; s < step+1; s++ {
		u := m0 + s
		g.AddVertex(u)

		vcount := len(g.Vertices()) - 1
		ecount := len(g.Edges())
		for mm := 1; mm < m+1; mm++ {
			total := A*float64(vcount) + float64(ecount)
			thresh := rand.Float64() * total
			var sum float64
			for v := 1; v < u; v++ {
				sum += A + float64(g.InDegree(v))
				if sum >= thresh {
					if mm == 1 {
						g.AddEdge(u, v)
					} else {
						g.AddEdge(rand.Intn(u)+1, v)
					}
					break
				}
			}
		}
	}
}

func (g *Graph) CreateLatentGraph(N, E int, errorRate float64, confer, dist string, alpha float64) {
}

func (g *Graph) CreateLatticeGraph(dim, n int, is_torus bool) {
	if dim == 1 {
		for i := 1; i < n+1; i++ {
			u := latticeVertex(dim, n, i)
			v := latticeVertex(dim, n, i+1)
			if is_torus || v > u {
				g.AddEdge(u, v)
			}
		}
	} else if dim == 2 {
		for j := 1; j < n+1; j++ {
			for i := 1; i < n+1; i++ {
				u := latticeVertex(dim, n, i, j)
				v := latticeVertex(dim, n, i+1, j)
				if is_torus || v > u {
					g.AddEdge(u, v)
				}
				v = latticeVertex(dim, n, i, j+1)
				if is_torus || v > u {
					g.AddEdge(u, v)
				}
			}
		}
	}
}

func latticeVertex(dim, n int, positions ...int) int {
	v := 0
	for _, i := range positions {
		v *= n
		if i > n {
			i -= n
		}
		if i < 1 {
			i += n
		}
		v += i - 1
	}
	return v + 1
}

func (g *Graph) CreateVoronoiGraph() {
}

func (g *Graph) CreateDegreeBoundedGraph(N, E int) {
}

func (g *Graph) CreateConfigurationGraph(degreeSeq []int) {
	g.ExpectDirected()

	var stubs []int
	for i, k := range degreeSeq {
		for j := 0; j < k; j++ {
			stubs = append(stubs, i+1)
		}
	}
	if len(stubs)%2 != 0 {
		die("Total degree must be even.")
	}

	N := len(degreeSeq)
	for {
		if g.ConnectRandomly(N, stubs) {
			break
		}
	}
}

func (g *Graph) ConnectRandomly(N int, stubs_ []int) bool {
	g.New()
	for v := 1; v < N+1; v++ {
		g.AddVertex(v)
	}

	stubs := stubs_
	rand.Shuffle(len(stubs), func(i, j int) { stubs[i], stubs[j] = stubs[j], stubs[i] })
	for len(stubs) > 0 {
		u := stubs[0]
		stubs = stubs[1:]
		ntries := 0
		for {
			v := stubs[0]
			stubs = stubs[1:]
			if u != v && !g.HasEdge(u, v) {
				g.AddEdge(u, v)
				break
			} else {
				stubs = append(stubs, v)
				ntries++
				if ntries > len(stubs) {
					return false
				}
			}
		}
	}
	return true
}

func (g *Graph) CreateRandomRegularGraph(N, k int) {
	degreeSeq := make([]int, N)
	for i := 0; i < N; i++ {
		degreeSeq[i] = k
	}
	g.CreateConfigurationGraph(degreeSeq)
}

func (g *Graph) CreateLiMainiGraph() {
}

func (g *Graph) ImportDot(lines []string) {
	var buf string
	for _, line := range lines {
		pos := strings.Index(line, "//")
		if pos >= 0 {
			line = line[pos:]
		}
		line = strings.TrimRight(line, "\n")
		buf += line
	}
	// remove C-style comment
	re := regexp.MustCompile(`/\*.*?\*/`)
	re.ReplaceAllString(buf, "")
	re = regexp.MustCompile(`graph\s+(\S+)\s*{(.*)}`)
	m := re.FindStringSubmatch(buf)
	body := m[2]
	g.importDotBody(body)
}

func (g *Graph) importDotBody(body string) {
	for _, line := range strings.Split(body, ";") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.Index(line, "graph") > -1 ||
			strings.Index(line, "node") > -1 ||
			strings.Index(line, "edge") > -1 {
			continue
		}
		re := regexp.MustCompile(`([^\[]*)\s*(\[(.*)\])?`)
		m := re.FindStringSubmatch(line)

		val, opts := m[1], m[3]
		val = strings.ReplaceAll(val, "\"", "")
		val = strings.TrimSpace(val)

		// parse attributes [name1=val1,name2=val2...]
		attrs := make(map[string]interface{})
		for _, pair := range strings.Split(opts, ",") {
			if pair == "" {
				break
			}
			pair = strings.TrimSpace(pair)
			k, v := strings.Split(pair, "=")[0], strings.Split(pair, "=")[1]
			v = strings.ReplaceAll(v, "\"", "")
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				attrs[k] = v
			} else {
				attrs[k] = f
			}
		}

		// parse vertex/edge definition
		// vertex -- vertex [-- vertex...]
		if strings.Index(val, "--") > -1 || strings.Index(val, "->") > -1 {
			re = regexp.MustCompile(`\s*-[->]\s*`)
			vertices := re.Split(val, -1)
			for i := 0; i < len(vertices)-1; i++ {
				u, _ := strconv.Atoi(vertices[i])
				v, _ := strconv.Atoi(vertices[i+1])
				g.AddEdge(u, v)
				id := g.EdgeCount(u, v) - 1
				g.SetEdgeAttributesByID(u, v, id, attrs)
			}
		} else {
			v, _ := strconv.Atoi(val)
			g.AddVertex(v)
			g.SetVertexAttributes(v, attrs)
		}
	}
}

func (g *Graph) ExportDot() string {
	str := g.headerString("// ")

	var head string
	if g.Directed() {
		head = "digraph"
	} else {
		head = "graph"
	}
	str += head + " export_dot {\n  node [color=gray90,style=filled];\n"

	V := g.Vertices()
	sort.Ints(V)
	for _, v := range V {
		str += fmt.Sprintf("  \"%d\"", v)
		var attrs []string
		for key, val := range g.VertexAttributes(v) {
			attrs = append(attrs, fmt.Sprintf("%v=\"%v\"", key, val))
		}
		if len(attrs) > 0 {
			str += " [" + strings.Join(attrs, ", ") + "]"
		}
		str += ";\n"
	}

	E := g.Edges()
	sort.Slice(E, func(i, j int) bool {
		if E[i][0] != E[j][0] {
			return E[i][0] < E[j][0]
		} else {
			return E[i][1] < E[j][1]
		}
	})
	for _, e := range E {
		u, v := e[0], e[1]
		if g.Undirected() && v < u {
			u, v = v, u
		}
		var l string
		if g.Directed() {
			l = "->"
		} else {
			l = "--"
		}
		str += fmt.Sprintf("  \"%d\" %s \"%d\"", u, l, v)
		for _, i := range g.MultiEdgeIDs(u, v) {
			var attrs []string
			for key, val := range g.EdgeAttributesByID(u, v, i) {
				attrs = append(attrs, fmt.Sprintf("%v=\"%v\"", key, val))
			}
			if len(attrs) > 0 {
				str += " [" + strings.Join(attrs, ", ") + "]"
			}
		}
		str += ";\n"
	}
	str += "}"
	return str
}

func (g *Graph) ExportCell() string {
	out := `#define v_size 4 4
#define v_color yellow
#define e_width 1
#define e_color blue
`

	V := g.Vertices()
	sort.Ints(V)
	for _, v := range V {
		out += fmt.Sprintf("define v%d ellipse v_size v_color\n", v)
	}

	E := g.Edges()
	sort.Slice(E, func(i, j int) bool { return E[i][0] < E[j][0] })
	for n, e := range E {
		u, v := e[0], e[1]
		out += fmt.Sprintf("define e%d link v%d v%d e_width e_color\n", n+1, u, v)
	}

	out += "spring /^v/\n"
	out += "display\n"
	out += "wait"

	return out
}

func (g *Graph) headerString(comment string) string {
	t := time.Now()
	time := t.Format(time.RFC1123)
	var directed string
	if g.Directed() {
		directed = "directed"
	} else {
		directed = "undirected"
	}
	vcount := len(g.Vertices())
	ecount := len(g.Edges())
	return fmt.Sprintf(`%sGenerated by graph-tools v%s at %s
%s%s, %d vertices, %d edges
`, comment, version, time, comment, directed, vcount, ecount)
}

func die(str string) {
	fmt.Println(str)
	os.Exit(1)
}
