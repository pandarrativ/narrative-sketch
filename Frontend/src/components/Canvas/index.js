import * as d3 from 'd3'
import { ReactPainter } from 'react-painter'
import { Input, Button } from 'antd'
import './index.scss'
import React, { useState } from 'react'
import MyImage from '../../common/libs/Image'

function clearCanvas() {
  var c = document.getElementById('canvas').children
  c[0].height = c[0].height
}

const DEFAULT_LINE_WIDTH = 10

class Canvas extends React.Component {
  constructor(props) {
    super(props)
    this.state = { width: 0, height: 0 }
    this.setLineWidth = () => {}
    this.setColor = () => {}
    this.canvasRef = React.createRef()
  }

  componentDidMount() {
    if (this.state.width === 0 && this.state.height === 0) {
      const paint = d3.select(this.canvasRef.current).select('.paint').node()
      const width = paint.clientWidth
      const height = paint.clientHeight
      this.setState({
        width,
        height
      })
    }
  }

  render() {
    return (
      <div className="canvas" ref={this.canvasRef}>
        <div className="paint">
          {this.state.width > 0 && this.state.height > 0 ? (
            <ReactPainter
              width={this.state.width}
              height={this.state.height}
              initialLineWidth={DEFAULT_LINE_WIDTH}
              onSave={async (blob) => {
                // const myImage = new MyImage()
                // await myImage.readFromBlob(blob)
                // this.props.setSketchingImage(myImage)
              }}
              image={this.props.image}
              render={({ canvas, setColor, setLineWidth, imageDownloadUrl }) => {
                this.setLineWidth = setLineWidth
                this.setColor = setColor
                return (
                  <div className="canv" id="canvas">
                    {canvas}
                  </div>
                )
              }}
            />
          ) : (
            <></>
          )}
          <div className="controller">
            <Button className="btn" type="primary" onClick={() => clearCanvas()} danger>
              Clear
            </Button>
            <input className="color" type="color" onChange={(e) => this.setColor(e.target.value)} />
            <label className="line">Line Width:</label>
            <Input
              type="number"
              className="line-input"
              placeholder="10"
              onChange={(e) => this.setLineWidth(e.target.value)}
            />
          </div>
        </div>
      </div>
    )
  }
}

export default Canvas
