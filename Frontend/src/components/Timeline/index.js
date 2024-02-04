import * as d3 from 'd3'
import React, { useEffect, useState } from 'react'
import 'antd/dist/antd.css'

import './index.scss'

const NODE_WIDTH = 100
const FONT_SIZE = 10
const NODE_HEIGHT = NODE_WIDTH + FONT_SIZE * 3
const EACH_TIMELINE_HEIGHT = 250

const Timeline = ({ data, id, style, onTimelineClick, comImgAdd, comType }) => {
  const [groups, setGroups] = useState([])
  const [mouseDownIndex, setMouseDownIndex] = useState(-1)

  const timelineRef = React.createRef()

  useEffect(() => {
    // x, y mapping
    const timeline = timelineRef.current
    const hasTimelineNodes = data.length > 0 && data[0].nodes.length > 0
    if (!!timeline && hasTimelineNodes) {
      const svg = d3.select(timeline).select('#storyline')
      const width = svg.node().clientWidth
      const margin = 10
      const left = margin + NODE_WIDTH / 2
      const right = width - (margin + NODE_WIDTH / 2)

      const top = margin + NODE_HEIGHT / 2
      const bottom = EACH_TIMELINE_HEIGHT - (margin + NODE_HEIGHT / 2)

      // first ensure the node position of the longest path
      const positions = {}
      let maxPosition = 4
      data.forEach((timeline) => {
        timeline.nodes.forEach((node, i) => {
          positions[node.id] = i
          maxPosition = Math.max(maxPosition, i)
        })
      })

      const x = d3.scaleLinear().domain([0, maxPosition]).range([left, right])
      const y = function (index, score) {
        const nodes = data[index].nodes
        const scoreScale = d3
          .scaleLinear()
          .domain([
            d3.max(nodes, (node) => node.score) + 1,
            d3.min(nodes, (node) => node.score) - 1
          ])
          .range([index * EACH_TIMELINE_HEIGHT + top, index * EACH_TIMELINE_HEIGHT + bottom])
        return scoreScale(score)
      }

      data.forEach((timeline, index) => {
        timeline.nodes.forEach((n) => {
          n.x = x(positions[n.id])
          n.y = y(index, n.score)
        })
      })

      // nodes
      const nodeMap = {} // id => node
      data.forEach((timeline, index) => {
        timeline.nodes.forEach((n) => {
          nodeMap[n.id] = n

          // n.timelineName = timeline.name
          // nodes.push(n)
        })
      })

      // groups
      const groups = []
      data.forEach((timeline, index) => {
        const group = {
          nodes: [],
          links: [],
          x: 0,
          y: index * EACH_TIMELINE_HEIGHT,
          width: width,
          height: EACH_TIMELINE_HEIGHT
        }
        timeline.nodes.forEach((n) => {
          n.timelineName = timeline.name
          group.nodes.push(n)
        })
        timeline.links.forEach((link) => {
          group.links.push({
            source: nodeMap[link.source],
            target: nodeMap[link.target]
          })
        })
        groups.push(group)
      })
      setGroups(groups)

      svg.attr('height', data.length * EACH_TIMELINE_HEIGHT)
    }
  }, [data])

  const colorArray = [
    '#4e79a7',
    '#f28e2c',
    '#e15759',
    '#76b7b2',
    '#59a14f',
    '#edc949',
    '#af7aa1',
    '#ff9da7',
    '#9c755f',
    '#bab0ab'
  ]

  return (
    <div
      id={id}
      style={{
        ...style,
        overflowX: 'hidden',
        overflowY: 'scroll'
      }}
      className="timeline"
      ref={timelineRef}
    >
      {data.length > 0 ? (
        <svg style={{ width: '100%' }} id="storyline">
          {groups.map((group, i) => (
            <g
              key={i}
              onTouchStart={() => {
                setMouseDownIndex(i)
              }}
              onTouchEnd={() => {
                setMouseDownIndex(-1)
                onTimelineClick(i)
              }}
            >
              <rect
                x={group.x}
                y={group.y}
                width={group.width}
                height={group.height}
                style={{ fill: mouseDownIndex === i ? '#ddd' : i % 2 ? 'transparent' : 'white' }}
              ></rect>
              <line
                x1={group.x}
                y1={group.y + group.height - 1}
                x2={group.x + group.width}
                y2={group.y + group.height - 1}
                style={{ stroke: 'gray', strokeDasharray: '10 5' }}
              ></line>
              {group.links.map((link) => (
                <path
                  key={`${link.source.id}-${link.target.id}`}
                  fill="none"
                  stroke={colorArray[i]}
                  strokeWidth={3}
                  d={d3
                    .linkHorizontal()
                    .source(function (d) {
                      return [d.source.x, d.source.y]
                    })
                    .target(function (d) {
                      return [d.target.x, d.target.y]
                    })(link)}
                ></path>
              ))}
              {group.nodes.map((node) => {
                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    className="timeline-node"
                  >
                    <rect
                      width={node.state === 'Combination' ? 2 * NODE_WIDTH : NODE_WIDTH}
                      height={NODE_HEIGHT}
                      x={-NODE_WIDTH / 2}
                      y={-NODE_HEIGHT / 2}
                      style={{
                        fill: 'white'
                      }}
                    ></rect>
                    <text
                      x={-NODE_WIDTH / 2 + 5}
                      y={-NODE_HEIGHT / 2 + FONT_SIZE + 5}
                      style={{ fontSize: FONT_SIZE * 1.2, fontWeight: 'bold' }}
                    >
                      {node.timelineName}
                    </text>
                    <text
                      x={-NODE_WIDTH / 2 + 5}
                      y={-NODE_HEIGHT / 2 + FONT_SIZE * 2.7}
                      style={{ fontSize: FONT_SIZE }}
                    >
                      {node.state}
                    </text>
                    <image
                      transform={`translate(${-NODE_WIDTH / 2}, ${
                        -NODE_WIDTH / 2 + (NODE_HEIGHT - NODE_WIDTH) / 2
                      })`}
                      width={NODE_WIDTH}
                      height={NODE_WIDTH}
                      href={node.AISketch}
                    ></image>
                    {node.state === 'Combination' ? (
                      <text
                        x={-NODE_WIDTH / 2 + 5 + NODE_WIDTH}
                        y={-NODE_HEIGHT / 2 + FONT_SIZE + 5}
                        style={{ fontSize: FONT_SIZE * 1.2, fontWeight: 'bold' }}
                      >
                        {comType}
                      </text>
                    ) : null}

                    {node.state === 'Combination' ? (
                      <image
                        transform={`translate(${NODE_WIDTH / 2}, ${
                          -NODE_WIDTH / 2 + (NODE_HEIGHT - NODE_WIDTH) / 2
                        })`}
                        width={NODE_WIDTH}
                        height={NODE_WIDTH}
                        href={comImgAdd}
                      ></image>
                    ) : null}

                    <rect
                      width={node.state === 'Combination' ? 2 * NODE_WIDTH : NODE_WIDTH}
                      height={NODE_HEIGHT}
                      x={-NODE_WIDTH / 2}
                      y={-NODE_HEIGHT / 2}
                      style={{
                        fill: 'none',
                        stroke: colorArray[i],
                        strokeWidth: 2
                      }}
                    ></rect>
                  </g>
                )
              })}
            </g>
          ))}
        </svg>
      ) : (
        <p className="timeline-placeholder">NO TIMELINE DATA YET</p>
      )}
    </div>
  )
}
export default Timeline
