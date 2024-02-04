var Graph = (function (undefined) {
  var extractKeys = function (obj) {
    var keys = [],
      key
    for (key in obj) {
      Object.prototype.hasOwnProperty.call(obj, key) && keys.push(key)
    }
    return keys
  }

  var sorter = function (a, b) {
    // return parseFloat(a) - parseFloat(b)
    return parseFloat(b) - parseFloat(a)
  }

  var findPaths = function (map, start, end, infinity) {
    infinity = infinity || Infinity

    var costs = {},
      open = { 0: [start] },
      predecessors = {},
      keys

    var addToOpen = function (cost, vertex) {
      var key = '' + cost
      if (!open[key]) open[key] = []
      open[key].push(vertex)
    }

    costs[start] = 0

    while (open) {
      if (!(keys = extractKeys(open)).length) break

      keys.sort(sorter)

      var key = keys[0],
        bucket = open[key],
        node = bucket.shift(),
        currentCost = parseFloat(key),
        adjacentNodes = map[node] || {}

      if (!bucket.length) delete open[key]

      for (var vertex in adjacentNodes) {
        if (Object.prototype.hasOwnProperty.call(adjacentNodes, vertex)) {
          var cost = adjacentNodes[vertex],
            totalCost = cost + currentCost,
            vertexCost = costs[vertex]

          if (vertexCost === undefined || vertexCost > totalCost) {
            costs[vertex] = totalCost
            addToOpen(totalCost, vertex)
            predecessors[vertex] = node
          }
        }
      }
    }

    if (costs[end] === undefined) {
      return null
    } else {
      return predecessors
    }
  }

  var extractLongest = function (predecessors, end) {
    var nodes = [],
      u = end

    while (u !== undefined) {
      nodes.push(u)
      u = predecessors[u]
    }

    nodes.reverse()
    return nodes
  }

  var findLongestPath = function (map, nodes) {
    var start = nodes.shift(),
      end,
      predecessors,
      path = [],
      longest

    while (nodes.length) {
      end = nodes.shift()
      predecessors = findPaths(map, start, end)

      if (predecessors) {
        longest = extractLongest(predecessors, end)
        if (nodes.length) {
          path.push.apply(path, longest.slice(0, -1))
        } else {
          return path.concat(longest)
        }
      } else {
        return null
      }

      start = end
    }
  }

  var toArray = function (list, offset) {
    try {
      return Array.prototype.slice.call(list, offset)
    } catch (e) {
      var a = []
      for (var i = offset || 0, l = list.length; i < l; ++i) {
        a.push(list[i])
      }
      return a
    }
  }

  var Graph = function (map) {
    this.map = map
  }

  Graph.prototype.findLongestPath = function (start, end) {
    if (Object.prototype.toString.call(start) === '[object Array]') {
      return findLongestPath(this.map, start)
    } else if (arguments.length === 2) {
      return findLongestPath(this.map, [start, end])
    } else {
      return findLongestPath(this.map, toArray(arguments))
    }
  }

  Graph.findLongestPath = function (map, start, end) {
    if (Object.prototype.toString.call(start) === '[object Array]') {
      return findLongestPath(map, start)
    } else if (arguments.length === 3) {
      return findLongestPath(map, [start, end])
    } else {
      return findLongestPath(map, toArray(arguments, 1))
    }
  }

  return Graph
})()

export default Graph
