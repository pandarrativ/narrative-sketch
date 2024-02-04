import React, { useEffect, useState, useRef } from 'react'
import { fabric } from 'fabric'
import './index.scss'

const CanvasCom = ({ selectImgAdd, comImgAdd }) => {
  //   const [can, setCan] = useState()
  const [canREF, _] = useState(React.createRef())
  let [canv, setCanv] = useState(undefined)
  let [imgset, setImgSet] = useState(new Set([]))

  //   console.log("selectImgAdd: \n", selectImgAdd)

  useEffect(() => {
    console.log('imgset: \n', imgset, 'has selectImgAdd: \n', imgset.has(selectImgAdd))
    if (canREF && canREF.current && !canv) {
      canv = new fabric.Canvas('c')
      setCanv(canv)
    }
    if (selectImgAdd && canv && !imgset.has(selectImgAdd)) {
      fabric.Image.fromURL(selectImgAdd, function (img) {
        canv.add(img)
        canv.renderAll()
      })
      imgset.add(selectImgAdd)
      setImgSet(new Set(imgset))
    }
    console.log('comImgAdd: \n', comImgAdd)
    if (comImgAdd !== '' && !imgset.has(comImgAdd)) {
      fabric.Image.fromURL(comImgAdd, function (img) {
        canv.add(img)
        canv.renderAll()
      })
      imgset.add(comImgAdd)
      setImgSet(new Set(imgset))
    }

    // setCan(canv)
  }, [canREF, selectImgAdd, comImgAdd, canv, imgset])

  return <canvas id="c" width={300} height={350} ref={canREF}></canvas>
}
export default CanvasCom
