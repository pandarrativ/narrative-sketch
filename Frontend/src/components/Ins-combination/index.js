import React, { useEffect, useState, useRef } from 'react'
import * as d3 from 'd3'
import { Button } from 'antd'
import { ArrowRightOutlined } from '@ant-design/icons'
import 'antd/dist/antd.css'
import './index.scss'
import { fetchRelevantObjects } from '../../common/libs/message'
import { link } from 'd3'

// Hook
function usePrevious(value) {
  // The ref object is a generic container whose current property is mutable ...
  // ... and can hold any value, similar to an instance property on a class
  const ref = useRef()
  // Store current value in ref
  useEffect(() => {
    ref.current = value
  }, [value]) // Only re-run if value changes
  // Return previous value (happens before update in useEffect above)
  return ref.current
}

const InsCombination = ({ data, setComImgAdd, setComType, selectImgAdd }) => {
  const [step, setStep] = useState(2)
  const [imgId, setImgId] = useState(-1)
  const [egoNetwork, setEgoNetwork] = useState()
  const [comImg, setComImg] = useState([])
  const [egoNetworkSVGRef, _] = useState(React.createRef())
  // var objImgPath = selectImgAdd

  useEffect(() => {
    const egoNetworkSVG = egoNetworkSVGRef.current
    if (!!egoNetworkSVG) {
      const egoWord = data.objName
      const relaventObjs = Object.keys(data.graph).map((key) => ({
        id: key,
        label: key,
        img: data.graph[key][0].imgPath, //图片网址
        contents: data.graph[key]
      }))
      const nodes = relaventObjs
      const egoNode = {
        id: egoWord,
        isEgo: true,
        label: egoWord,
        img: selectImgAdd
      }
      const links = []
      nodes.forEach((node) => {
        links.push({
          source: egoNode.id,
          target: node.id
        })
      })
      nodes.push(egoNode)

      setEgoNetwork({ nodes, links })
    }
  }, [egoNetworkSVGRef, data])

  useEffect(() => {
    const egoNetworkSVG = egoNetworkSVGRef.current
    if (!!egoNetworkSVG && egoNetwork?.nodes?.length > 0) {
      const size = 1
      const width = egoNetworkSVG.clientWidth * size
      const height = egoNetworkSVG.clientHeight * size
      const center = { x: width / 2, y: height / 2 }
      const BASIC_RADIUS = 1
      const DISTANCE_RADIUS_RATIO = 3
      const BASIC_DISTANCE = BASIC_RADIUS * DISTANCE_RADIUS_RATIO
      const FONT_SIZE = 15

      const egoNode = egoNetwork.nodes.filter((n) => n.isEgo)[0]
      egoNode.x = center.x
      egoNode.y = center.y
      egoNode.r = BASIC_RADIUS
      egoNode.fontSize = FONT_SIZE
      egoNode.tx = egoNode.x + egoNode.r // text x
      egoNode.ty = egoNode.y + egoNode.fontSize / 2 // egoNode.y + egoNode.r + egoNode.fontSize // text y

      const nodes = egoNetwork.nodes.filter((n) => !n.isEgo)
      nodes.forEach((node, i) => {
        const theta = (i * 2 * Math.PI) / nodes.length
        node.x = Math.sin(theta) * BASIC_DISTANCE + egoNode.x
        node.y = Math.cos(theta) * BASIC_DISTANCE + egoNode.y
        node.fontSize = FONT_SIZE
        node.r = BASIC_RADIUS
        node.tx = node.x + node.r // Math.sin(theta) * (BASIC_DISTANCE + node.r) + egoNode.x
        node.ty = node.y + node.fontSize / 4 // Math.cos(theta) * (BASIC_DISTANCE + node.r) + egoNode.y
      })

      const svg = d3.select(egoNetworkSVG)
      let linkG = svg.select('g.links')
      linkG = linkG.empty() ? svg.append('g').classed('links', true) : linkG
      let textG = svg.select('g.texts')
      textG = textG.empty() ? svg.append('g').classed('texts', true) : textG
      let nodeG = svg.select('g.nodes')
      nodeG = nodeG.empty() ? svg.append('g').classed('nodes', true) : nodeG

      const textSelection = textG.selectAll('text.label').data(egoNetwork.nodes)
      textSelection.enter().append('text').classed('label', true)
      textSelection.exit().remove()
      textG
        .selectAll('text.label')
        .attr('x', (d) => d.tx)
        .attr('y', (d) => d.ty)
        .attr('text-anchor', (d) => d.textAnchor)
        .attr('font-size', (d) => d.fontSize)
        .text((d) => d.label)

      const nodeSelection = nodeG.selectAll('g.node').data(egoNetwork.nodes)
      nodeSelection.enter().append('g').classed('node', true)
      nodeSelection.exit().remove()
      nodeG
        .selectAll('g.node')
        .attr('transform', (d) => `translate(${d.x}, ${d.y})`)
        .each(function (d) {
          const g = d3.select(this)
          let image = g.select('image')
          image = image.empty() ? g.append('image') : image
          image
            .attr('transform', `translate(${-d.r}, ${-d.r})`)
            .attr('width', d.r * 2)
            .attr('height', d.r * 2)
            .attr('href', d.img)
          let circle = g.select('circle')
          circle = circle.empty() ? g.append('circle') : circle
          circle
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', d.r)
            .style('fill', 'none')
            .style('stroke', 'black')
            .style('stroke-width', 3)
        })

      const nodeGBBox = nodeG.node().getBBox()
      const textGBBox = textG.node().getBBox()
      const ratio = Math.min(
        (width / 2 - (nodeGBBox.x - textGBBox.x)) / (width / 2 - nodeGBBox.x), // left
        (height / 2 - (nodeGBBox.y - textGBBox.y)) / (height / 2 - nodeGBBox.y), // top
        (width / 2 - (textGBBox.x + textGBBox.width - (nodeGBBox.x + nodeGBBox.width))) /
          (nodeGBBox.x + nodeGBBox.width - width / 2), // right
        (height / 2 - (textGBBox.y + textGBBox.height - (nodeGBBox.y + nodeGBBox.height))) /
          (nodeGBBox.y + nodeGBBox.height - height / 2) // bottom
      )
      // console.log(ratio)
      // const ratio = Math.min(
      //   (width - textGBBox.width) / nodeGBBox.width,
      //   (height - textGBBox.height) / nodeGBBox.height
      // )

      const distance = BASIC_DISTANCE * ratio
      const radius = BASIC_RADIUS * ratio
      nodes.forEach((node, i) => {
        const theta = (i * 2 * Math.PI) / nodes.length
        node.x = Math.sin(theta) * distance + egoNode.x
        node.y = Math.cos(theta) * distance + egoNode.y
        node.r = radius
        node.tx = node.x + node.r * 1.1 // Math.sin(theta) * (distance + node.r) + egoNode.x
        node.ty = node.y + node.fontSize / 4 // Math.cos(theta) * (distance + node.r) + egoNode.y
      })
      egoNode.r = radius
      egoNode.tx = egoNode.x + egoNode.r * 1.1
      egoNode.ty = egoNode.y + egoNode.fontSize / 4 // egoNode.y + egoNode.r + egoNode.fontSize // text y

      // link
      const LINK_COLOR = 'gray'
      let defs = svg.select('defs')
      if (defs.empty()) {
        defs = svg.append('defs')

        defs
          .append('marker')
          .attr('id', 'arrow')
          .attr('markerUnits', 'strokeWidth')
          .attr('markerWidth', '12')
          .attr('markerHeight', '12')
          .attr('viewBox', '0 0 12 12')
          .attr('refX', '10')
          .attr('refY', '6')
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M2,2 L10,6 L2,10 L6,6 L2,2')
          .attr('style', `fill: ${LINK_COLOR};`)

        defs
          .append('clipPath')
          .attr('id', 'circleView')
          .append('circle')
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('r', radius)
          .attr('fill', '#FFFFFF')
      }

      const id2node = egoNetwork.nodes.reduce((map, node) => {
        map[node.id] = node
        return map
      }, {})
      const cutLink = (d) => {
        const source = id2node[d.source]
        const target = id2node[d.target]
        let x1 = source.x
        let y1 = source.y
        let x2 = target.x
        let y2 = target.y
        const distance = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
        if (y1 === y2) {
          x1 += (source.r * (y2 - y1)) / Math.abs(y2 - y1)
          x2 -= (target.r * (y2 - y1)) / Math.abs(y2 - y1)
        } else {
          x1 += (source.r / distance) * (x2 - x1)
          y1 += (source.r / distance) * (y2 - y1)
          x2 -= (target.r / distance) * (x2 - x1)
          y2 -= (target.r / distance) * (y2 - y1)
        }
        return { x1, x2, y1, y2 }
      }
      const linkSelection = linkG.selectAll('line.link').data(egoNetwork.links.map(cutLink))
      linkSelection.enter().append('line').classed('link', true)
      linkSelection.exit().remove()
      linkG
        .selectAll('line.link')
        .attr('x1', (d) => d.x1)
        .attr('y1', (d) => d.y1)
        .attr('x2', (d) => d.x2)
        .attr('y2', (d) => d.y2)
        .attr('stroke', LINK_COLOR)
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrow)')

      nodeG
        .selectAll('g.node')
        .attr('transform', (d) => `translate(${d.x}, ${d.y})`)
        .style('clip-path', 'url(#circleView)')
        .each(function (d) {
          // console.log(d)
          const g = d3.select(this)
          g.on('click', () => {
            // console.log('g: \n', g._groups[0][0].__data__.id)
            setComType(g._groups[0][0].__data__.id)
            setComImg(g._groups[0][0].__data__.contents)
          })
          let image = g.select('image')
          image = image.empty() ? g.append('image') : image
          image
            .attr('transform', `translate(${-d.r}, ${-d.r})`)
            .attr('width', d.r * 2)
            .attr('height', d.r * 2)
            .attr('href', d.img)
          let circle = g.select('circle')
          circle = circle.empty() ? g.append('circle') : circle
          circle
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', d.r)
            .style('fill', 'none')
            .style('stroke', 'black')
            .style('stroke-width', 3)
        })

      textG
        .selectAll('text.label')
        .attr('x', (d) => d.tx)
        .attr('y', (d) => d.ty)
        .attr('font-size', (d) => d.fontSize)
    }
  }, [egoNetwork, egoNetworkSVGRef])

  // console.log('comImg: \n', comImg)

  return (
    <div className="combination">
      {/* <div className="combination-top">
        <div>Inspiration-combination</div>
      </div> */}
      <div className="combination-bottom">
        {step >= 0 && (
          <div className="step graph">
            <svg className="ego-network" ref={egoNetworkSVGRef}></svg>
          </div>
        )}
        {step >= 1 && (
          <div className="arrow">
            <ArrowRightOutlined />
          </div>
        )}
        {step >= 1 && (
          <div className="step picture">
            {comImg.map((info, index) => (
              <img
                className={`comimg${imgId === index ? ' chosen' : ''}`}
                onClick={() => {
                  setImgId(index)
                  setComImgAdd(info['imgPath'])
                  // setChosedImage(info['imgPath'])
                }}
                src={info['imgPath']}
                key={index}
                alt={index}
              />
            ))}
          </div>
        )}
      </div>
      {/* {step >= 2 && (
        <div className="arrow">
          <ArrowRightOutlined />
        </div>
      )}
      {step >= 2 && <div className="step position"></div>} */}
    </div>
  )
}
export default InsCombination
