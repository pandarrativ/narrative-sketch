import React, { useEffect, useRef, useState } from 'react'
import { Button, Input, Slider, Switch, Spin } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'
import 'antd/dist/antd.css'
import './index.scss'
import { images } from '../../assets'
import axios from 'axios'

const antIcon = (
  <LoadingOutlined
    style={{
      fontSize: 24
    }}
    spin
  />
)

const InsExploration = ({
  setMenu,
  exp_data,
  setExpImg,
  setSexp,
  setExpInputReason,
  exp_input_reason,
  isExplord
}) => {
  const [val0, setVal0] = useState(0)
  const [val1, setVal1] = useState(0)
  const [val2, setVal2] = useState(0)
  const [val3, setVal3] = useState(0)
  const [val4, setVal4] = useState(0)
  const [check0, setCheck0] = useState(false) //switch 开关当前状态
  const [way0Choose, setWay0Choose] = useState(false)
  const [len0, setLen0] = useState(1)
  const [check1, setCheck1] = useState(false) //switch 开关当前状态
  const [way1Choose, setWay1Choose] = useState(false)
  const [len1, setLen1] = useState(1)
  const [check2, setCheck2] = useState(false) //switch 开关当前状态
  const [way2Choose, setWay2Choose] = useState(false)
  const [len2, setLen2] = useState(1)
  const [check3, setCheck3] = useState(false) //switch 开关当前状态
  const [way3Choose, setWay3Choose] = useState(false)
  const [len3, setLen3] = useState(1)
  const [check4, setCheck4] = useState(false) //switch 开关当前状态
  const [way4Choose, setWay4Choose] = useState(false)
  const [len4, setLen4] = useState(1)
  const [inputName, setInputName] = useState('')
  const intervalId = useRef()

  const onChangeS0 = (value) => {
    //slider 0的onChange函数
    setSexp(value)
    setVal0(value)
  }
  const onChangeB0 = () => {
    //switch button 0的onChange函数
    if (check0 === false) {
      setCheck0(true)
      setWay0Choose(true)
      intervalId.current = setInterval(() => {
        axios.post('/api/ExpRefreshDirection', { directionName: 'pink' }).then((res) => {
          // console.log(res)
          const Exp_Img = res.data.sketchList
          const temp = []
          Exp_Img.forEach((element) => {
            temp.push({ imgPath: element.imgPath, imgName: element.sketchName })
          })
          setExpImg(temp)
          setLen0(Exp_Img.length)
        })
      }, 2000)
    } else if (check0 === true) {
      setCheck0(false)
      setWay0Choose(false)
      clearInterval(intervalId.current)
    }
  }

  const onChangeS1 = (value) => {
    //slider 1的onChange函数
    setSexp(value)
    setVal1(value)
  }
  const onChangeB1 = () => {
    //switch button 1的onChange函数
    if (check1 === false) {
      setCheck1(true)
      setWay1Choose(true)
      intervalId.current = setInterval(() => {
        axios.post('/api/ExpRefreshDirection', { directionName: 'happy' }).then((res) => {
          // console.log(res)
          const Exp_Img = res.data.sketchList
          const temp = []
          Exp_Img.forEach((element) => {
            temp.push({ imgPath: element.imgPath, imgName: element.sketchName })
          })
          setExpImg(temp)
          setLen1(Exp_Img.length)
        })
      }, 2000)
    } else if (check1 === true) {
      setCheck1(false)
      setWay1Choose(false)
      clearInterval(intervalId.current)
    }
  }

  const onChangeS2 = (value) => {
    //slider 2的onChange函数
    setSexp(value)
    setVal2(value)
  }
  const onChangeB2 = () => {
    //switch button 2的onChange函数
    if (check2 === false) {
      setCheck2(true)
      setWay2Choose(true)
      intervalId.current = setInterval(() => {
        axios.post('/api/ExpRefreshDirection', { directionName: 'fat' }).then((res) => {
          // console.log(res)
          const Exp_Img = res.data.sketchList
          const temp = []
          Exp_Img.forEach((element) => {
            temp.push({ imgPath: element.imgPath, imgName: element.sketchName })
          })
          setExpImg(temp)
          setLen2(Exp_Img.length)
        })
      }, 2000)
    } else if (check2 === true) {
      setCheck2(false)
      setWay2Choose(false)
      clearInterval(intervalId.current)
    }
  }

  const onChangeS3 = (value) => {
    //slider 3的onChange函数
    setSexp(value)
    setVal3(value)
  }
  const onChangeB3 = () => {
    //switch button 3的onChange函数
    if (check3 === false) {
      setCheck3(true)
      setWay3Choose(true)
      intervalId.current = setInterval(() => {
        axios.post('/api/ExpRefreshDirection', { directionName: 'shiny' }).then((res) => {
          // console.log(res)
          const Exp_Img = res.data.sketchList
          const temp = []
          Exp_Img.forEach((element) => {
            temp.push({ imgPath: element.imgPath, imgName: element.sketchName })
          })
          setExpImg(temp)
          setLen3(Exp_Img.length)
        })
      }, 2000)
    } else if (check3 === true) {
      setCheck3(false)
      setWay3Choose(false)
      clearInterval(intervalId.current)
    }
  }

  const onChangeS4 = (value) => {
    //slider 4的onChange函数
    setSexp(value)
    setVal4(value)
  }
  const onChangeB4 = () => {
    //switch button 4的onChange函数
    if (check4 === false) {
      setCheck4(true)
      setWay4Choose(true)
      intervalId.current = setInterval(() => {
        axios.post('/api/ExpRefreshDirection', { directionName: inputName }).then((res) => {
          setExpInputReason(res.data.reason)
          const Exp_Img = res.data.sketchList
          const temp = []
          Exp_Img.forEach((element) => {
            temp.push({ imgPath: element.imgPath, imgName: element.sketchName })
          })
          setExpImg(temp)
          setLen4(Exp_Img.length)
        })
      }, 2000)
    } else if (check4 === true) {
      setCheck4(false)
      setWay4Choose(false)
      clearInterval(intervalId.current)
    }
  }

  const onPressEnter = (e) => {
    setInputName(e.target.value)
  }

  // console.log(inputName)

  return (
    <>
      <div className="exploration-board-bottom">
        <div className="exploration-top">
          <div>Inspiration-exploration</div>
          <Spin indicator={antIcon} tip="lording" spinning={isExplord} />
        </div>
        <div
          className={`exploration-changeway${
            way0Choose === true ? ' exploration-changeway-selected' : ''
          }`}
        >
          <Switch className="switch" size="small" checked={check0} onChange={onChangeB0} />
          <Slider
            className="slider"
            tipFormatter={null}
            value={val0}
            min={0}
            max={len0 - 1}
            onChange={onChangeS0}
          />
          <div className="description">
            {exp_data.objName} becomes&ensp;
            <span className="name">
              {exp_data.textDirectionList ? exp_data.textDirectionList[0] : ''}
            </span>
            &ensp;because&thinsp;
            {exp_data.reasonList ? exp_data.reasonList[0] : ''}
          </div>
        </div>

        <div
          className={`exploration-changeway${
            way1Choose === true ? ' exploration-changeway-selected' : ''
          }`}
        >
          <Switch className="switch" size="small" checked={check1} onChange={onChangeB1} />
          <Slider
            className="slider"
            tipFormatter={null}
            value={val1}
            min={0}
            max={len1 - 1}
            onChange={onChangeS1}
          />
          <div className="description">
            {exp_data.objName} becomes&ensp;
            <span className="name">
              {exp_data.textDirectionList ? exp_data.textDirectionList[1] : ''}
            </span>
            &ensp;because&thinsp;
            {exp_data.reasonList ? exp_data.reasonList[1] : ''}
          </div>
        </div>

        <div
          className={`exploration-changeway${
            way2Choose === true ? ' exploration-changeway-selected' : ''
          }`}
        >
          <Switch className="switch" size="small" checked={check2} onChange={onChangeB2} />
          <Slider
            className="slider"
            tipFormatter={null}
            value={val2}
            min={0}
            max={len2 - 1}
            onChange={onChangeS2}
          />
          <div className="description">
            {exp_data.objName} becomes&ensp;
            <span className="name">
              {exp_data.textDirectionList ? exp_data.textDirectionList[2] : ''}
            </span>
            &ensp;because&thinsp;
            {exp_data.reasonList ? exp_data.reasonList[2] : ''}
          </div>
        </div>

        <div
          className={`exploration-changeway${
            way3Choose === true ? ' exploration-changeway-selected' : ''
          }`}
        >
          <Switch className="switch" size="small" checked={check3} onChange={onChangeB3} />
          <Slider
            className="slider"
            tipFormatter={null}
            value={val3}
            min={0}
            max={len3 - 1}
            onChange={onChangeS3}
          />
          <div className="description">
            {exp_data.objName} becomes&ensp;
            <span className="name">
              {exp_data.textDirectionList ? exp_data.textDirectionList[3] : ''}
            </span>
            &ensp;because&thinsp;
            {exp_data.reasonList ? exp_data.reasonList[3] : ''}
          </div>
        </div>

        <div
          className={`exploration-changeway${
            way4Choose === true ? ' exploration-changeway-selected' : ''
          }`}
        >
          <Switch className="switch" size="small" checked={check4} onChange={onChangeB4} />
          <Slider
            className="slider"
            tipFormatter={null}
            value={val4}
            min={0}
            max={len4 - 1}
            onChange={onChangeS4}
          />
          <div className="description">{exp_data.objName} becomes</div>
          <Input
            className="input"
            placeholder="Input a type and press Enter"
            onPressEnter={onPressEnter}
          />
          <div className="description-input">because {exp_input_reason}</div>
        </div>
      </div>
    </>
  )
}
export default InsExploration
