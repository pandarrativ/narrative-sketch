import './index.scss'
import { images } from '../../assets'
import React, { useEffect, useState } from 'react'
import { Button } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'
import {
  EXPLORATION,
  INCUBATION,
  PREPARATION,
  STATES,
  TIMELINE,
  TRANSFORMATION,
  COMBINATION
} from '../Container'
import CanvasCom from '../CanvasCom'

const BoxPI = ({
  step,
  setStep,
  disabled,
  inc_data,
  setImgName,
  selectImgAdd,
  setImgAdd,
  setGetSrc
}) => {
  const [imgSrc, setImgSrc] = useState('')
  // if (!inc_data?.length) debugger
  if (inc_data.length !== 0) {
    setImgName(inc_data[step]['sketchName'])
  }
  useEffect(() => {
    const imgAdd = inc_data?.length === 0 ? selectImgAdd : inc_data[step]['imgPath']
    setImgSrc(imgAdd)
    setImgAdd(imgAdd)
  }, [inc_data, step, imgSrc, setGetSrc, selectImgAdd, setImgAdd])
  return (
    <div id="AIDisplay">
      <img className="img" src={imgSrc} alt="" />
      <div className="controller">
        <Button
          id="add"
          onClick={() => {
            setStep(step + 1)
          }}
          disabled={disabled}
          type="primary"
        >
          +
        </Button>
        <Button
          id="minus"
          onClick={() => {
            setStep(step - 1)
          }}
          type="primary"
          disabled={disabled}
        >
          -
        </Button>
        <Button
          id="clear"
          type="primary"
          danger
          disabled={disabled}
          onClick={() => {
            setStep(0)
          }}
        >
          Clear
        </Button>
      </div>
    </div>
  )
}

const BoxExploration = ({ s_exp, exp_img, setImgName, setImgAdd }) => {
  // console.log('exp_img: \n', exp_img)
  if (exp_img.length !== 0) {
    setImgName(exp_img[s_exp]['imgName'])
    setImgAdd(exp_img[s_exp]['imgPath'])
  }
  return (
    <img
      className="change-img"
      src={
        exp_img.length === 0 ? images(`./images/white_background.png`) : exp_img[s_exp]['imgPath']
      }
      alt=""
    />
  )
}

const BoxTransformation = ({ s_tra, tra_data, setImgName, setImgAdd }) => {
  var temp = Object.keys(tra_data)
  // console.log('tra: \n', tra_data)
  if (temp.length !== 0) {
    setImgName(tra_data?.sketchList[s_tra]['sketchName'])
    setImgAdd(tra_data?.sketchList[s_tra]['imgPath'])
  }
  // console.log("temp: \n", temp)
  return (
    <img
      className="change-img"
      src={
        temp.length === 0
          ? images(`./images/white_background.png`)
          : tra_data?.sketchList[s_tra]['imgPath']
      }
      alt=""
    />
  )
}

const BoxCombination = ({ selectImgAdd, comImgAdd }) => {
  return (
    <div>
      <CanvasCom selectImgAdd={selectImgAdd} comImgAdd={comImgAdd} />
      {/* <img className="change-img-com" src={selectImgAdd} alt="" /> */}
      {/* <img className="change-img-com" src={comImgAdd === '' ? null : comImgAdd} alt="" /> */}
    </div>
  )
}

const AIDisplay = ({
  step,
  setStep,
  state,
  tra_data,
  inc_data,
  exp_img,
  setImgName,
  s_exp,
  s_tra,
  selectImgAdd,
  setImgAdd,
  islordingb,
  setGetSrc,
  comImgAdd
}) => {
  if (state === PREPARATION || state === INCUBATION || state === TIMELINE) {
    return (
      <BoxPI
        step={step}
        setStep={setStep}
        disabled={state === PREPARATION || state === TIMELINE}
        inc_data={inc_data}
        setImgName={setImgName}
        selectImgAdd={selectImgAdd}
        islordingb={islordingb}
        setImgAdd={setImgAdd}
        setGetSrc={setGetSrc}
      />
    )
  } else if (state === EXPLORATION) {
    return (
      <BoxExploration
        s_exp={s_exp}
        exp_img={exp_img}
        setImgName={setImgName}
        setImgAdd={setImgAdd}
      />
    )
  } else if (state === TRANSFORMATION) {
    return (
      <BoxTransformation
        s_tra={s_tra}
        tra_data={tra_data}
        setImgName={setImgName}
        setImgAdd={setImgAdd}
      />
    )
  } else if (state === COMBINATION) {
    return <BoxCombination selectImgAdd={selectImgAdd} comImgAdd={comImgAdd} />
  }
}

export default AIDisplay
