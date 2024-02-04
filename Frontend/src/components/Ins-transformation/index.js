import React, { useEffect, useState, useRef } from 'react'
import { Button, Slider, Input, Switch } from 'antd'
import './index.scss'
import axios from 'axios'

const InsTransformation = ({ setMenu, setStra, setTraData }) => {
  const [s, setS] = useState(0)
  const [flag, setFlag] = useState(-1)
  const [data, setData] = useState({})
  const [check0, setCheck0] = useState(false)
  const [check1, setCheck1] = useState(false)
  const [check2, setCheck2] = useState(false)
  const [check3, setCheck3] = useState(false)
  const [check4, setCheck4] = useState(false)
  const [len, setLen] = useState(1)
  const [inputName, setInputName] = useState('your type')
  const intervalId = useRef()

  var img_num = []
  if (data.sketchList?.length > 0) {
    const len = data.sketchList?.length
    for (let i = 0; i < len; i = i + 7) {
      img_num.push(i)
    }
    img_num.push(len - 1)
  }

  const onChangeB0 = () => {
    //switch button 0的onChange函数
    if (check0 === false) {
      setCheck0(true)
      setFlag(0)
      if (flag !== 0) {
        img_num = []
      }
      intervalId.current = setInterval(() => {
        axios.post('/api/TransRefreshDirection', { directionName: 'Devil' }).then((res) => {
          // console.log(res.data)
          setData(res.data)
          setTraData(res.data)
          setLen(res.data.sketchList?.length)
        })
      }, 2000)
    } else if (check0 === true) {
      setCheck0(false)
      clearInterval(intervalId.current)
      img_num = []
    }
  }

  const onChangeB1 = () => {
    //switch button 0的onChange函数
    if (check1 === false) {
      setCheck1(true)
      setFlag(1)
      if (flag !== 1) {
        img_num = []
      }
      intervalId.current = setInterval(() => {
        axios.post('/api/TransRefreshDirection', { directionName: 'Sun' }).then((res) => {
          // console.log(res.data)
          setData(res.data)
          setTraData(res.data)
          setLen(res.data.sketchList?.length)
        })
      }, 3000)
    } else if (check1 === true) {
      setCheck1(false)
      clearInterval(intervalId.current)
      img_num = []
    }
  }

  const onChangeB2 = () => {
    //switch button 0的onChange函数
    if (check2 === false) {
      setCheck2(true)
      setFlag(2)
      if (flag !== 2) {
        img_num = []
      }
      intervalId.current = setInterval(() => {
        axios.post('/api/TransRefreshDirection', { directionName: 'Angel' }).then((res) => {
          // console.log(res.data)
          setData(res.data)
          setTraData(res.data)
          setLen(res.data.sketchList?.length)
        })
      }, 3000)
    } else if (check2 === true) {
      setCheck2(false)
      clearInterval(intervalId.current)
      img_num = []
    }
  }

  const onChangeB3 = () => {
    //switch button 0的onChange函数
    if (check3 === false) {
      setCheck3(true)
      setFlag(3)
      if (flag !== 3) {
        img_num = []
      }
      intervalId.current = setInterval(() => {
        axios.post('/api/TransRefreshDirection', { directionName: 'Superman' }).then((res) => {
          // console.log(res.data)
          setData(res.data)
          setTraData(res.data)
          setLen(res.data.sketchList?.length)
        })
      }, 3000)
    } else if (check3 === true) {
      setCheck3(false)
      clearInterval(intervalId.current)
      img_num = []
    }
  }

  const onChange = (value) => {
    setS(value)
    setStra(value)
  }

  const onPressEnter = (e) => {
    setInputName(e.target.value)
  }
  const onChangeB4 = () => {
    //switch button 0的onChange函数
    if (check4 === false) {
      setCheck4(true)
      setFlag(4)
      if (flag !== 4) {
        img_num = []
      }
      intervalId.current = setInterval(() => {
        axios.post('/api/TransRefreshDirection', { directionName: inputName }).then((res) => {
          // console.log(res.data)
          setData(res.data)
          setTraData(res.data)
          setLen(res.data.sketchList?.length)
        })
      }, 3000)
    } else if (check4 === true) {
      setCheck4(false)
      clearInterval(intervalId.current)
      img_num = []
    }
  }

  return (
    <>
      <div className="transformation-board-bottom">
        <div className="transformation-top">
          <div>Inspiration-transformation</div>
        </div>

        <div className="transformation-changeway">
          <div className="changeway-left">
            <Input
              className="input"
              placeholder="Input a type and press Enter"
              onPressEnter={onPressEnter}
            />
            <Switch
              className="switch"
              checkedChildren={inputName}
              unCheckedChildren={inputName}
              checked={check4}
              onChange={onChangeB4}
            />
            <Switch
              className="switch"
              checkedChildren="Devil"
              unCheckedChildren="Devil"
              checked={check0}
              onChange={onChangeB0}
            />
            <Switch
              className="switch"
              checkedChildren="Sun"
              unCheckedChildren="Sun"
              checked={check1}
              onChange={onChangeB1}
            />
            <Switch
              className="switch"
              checkedChildren="Angel"
              unCheckedChildren="Angel"
              checked={check2}
              onChange={onChangeB2}
            />
            <Switch
              className="switch"
              checkedChildren="Superman"
              unCheckedChildren="Superman"
              checked={check3}
              onChange={onChangeB3}
            />
          </div>

          <div className="changeway-right">
            {flag === -1 ? null : (
              <>
                <div className="description">
                  {data?.objName} transforms into{' '}
                  <span className="name">{data?.directionName}</span> because {data?.reason}
                </div>
                <div className="showimg">
                  {img_num.map((num) => (
                    <img
                      className="showimg-img"
                      src={data?.sketchList[num]['imgPath']}
                      key={num}
                      alt={num}
                    />
                  ))}
                </div>
                <Slider className="slider" value={s} min={0} max={len - 1} onChange={onChange} />
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
export default InsTransformation
