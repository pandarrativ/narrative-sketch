import React, { useEffect, useState } from 'react'
import { Slider, Button, Spin } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'
import 'antd/dist/antd.css'
import './index.scss'
import { images } from '../../assets'

const antIcon = (
  <LoadingOutlined
    style={{
      fontSize: 24
    }}
    spin
  />
)

const Incubation = ({ step, setStep, inc_data, isInclord }) => {
  const img_num = []
  // console.log(inc_data)
  if (inc_data?.length > 0) {
    const t = Math.floor(inc_data?.length / 8)
    //  console.log("t: \n", t)
    for (let i = 0; i < inc_data?.length - t; i = i + t) {
      img_num.push(i)
    }
    img_num.push(inc_data?.length - 1)
  }

  // console.log("s:\n", s)

  const onChange = (value) => {
    setStep(value)
  }
  // console.log(img_num)

  return (
    <>
      <div className="incubation-board-bottom">
        <div className="incubation-top">
          <div>Incubation</div>
          <Spin indicator={antIcon} tip="lording" spinning={isInclord} />
        </div>
        <div className="showimg">
          {typeof inc_data === 'undefined' || inc_data?.length === 0 ? (
            <img className="showimg-img" src={images(`./images/white_background.png`)} alt="" />
          ) : (
            img_num.map((num) => (
              <img className="showimg-img" src={inc_data[num]['imgPath']} key={num} alt={num} />
            ))
          )}
        </div>
        <Slider
          className="slider"
          // tipFormatter={null}
          value={step}
          min={0}
          max={inc_data?.length - 1}
          onChange={onChange}
        />
      </div>
    </>
  )
}
export default Incubation
