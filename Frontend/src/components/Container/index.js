import { useEffect, useState, useRef } from 'react'
import { Popover, Slider, Button, message, InputNumber } from 'antd'
import {
  SaveOutlined,
  DeleteOutlined,
  VerticalAlignBottomOutlined,
  VerticalAlignTopOutlined,
  CheckOutlined
} from '@ant-design/icons'

import Canvas from '../Canvas'
import { ChooseBox } from '../preparations'
import Timeline from '../Timeline'
import Incubation from '..//Incubation'
import InsCombination from '../Ins-combination'
import InsExploration from '../Ins-exploration'
import InsTransformation from '../Ins-transformation'
import './index.scss'
import {
  pre,
  evaluation,
  incubation,
  ins_combination,
  ins_exploration,
  ins_transformation,
  timeline,
  finish
} from '../../assets'
import AIDisplay from '../AIDisplay'
import axios from 'axios'

export const [PREPARATION, INCUBATION, COMBINATION, EXPLORATION, TRANSFORMATION, TIMELINE, FINISH] =
  [
    'Preparation',
    'Incubation',
    'Combination',
    'Exploration',
    'Transformation',
    'Timeline',
    'Finish'
  ]
const DUMMY = 'Dummy'
const NONE_STATE = 'NoneState'
export const STATES = [
  PREPARATION,
  INCUBATION,
  COMBINATION,
  EXPLORATION,
  TRANSFORMATION,
  TIMELINE,
  FINISH
]

const Container = () => {
  const [menu, setMenu] = useState(0)
  const [page, setPage] = useState(<></>)
  const [step, setStep] = useState(0)
  const [s_exp, setSexp] = useState(0) //用于exploration的slider
  const [s_tra, setStra] = useState(0) //用于transformation的slider
  const [firstStep, setFirstStep] = useState(0)
  const [lastStep, setLastStep] = useState(1)
  const [imageSteps, setImageSteps] = useState(0)

  // timeline
  let [currentTimelineIndex, setCurrentTimelineIndex] = useState(-1)
  const [timelineData, setTimelineData] = useState([]) // several timelines, each timeline: nodes and links
  const [score, setScore] = useState(60)
  const [nodeType, setNodeType] = useState('')
  const [storeTooltipVisible, setStoreTooltipVisible] = useState(false)

  const [newStateFlag, setNewStateFlag] = useState(0)
  const [nowStateName, setNowStateName] = useState('') //当前state name
  const [com_data, setComData] = useState({
    graph: {
      flower: [
        {
          imgPath: 'https://hz-4.matpool.com:26142/static/ImgPool/flowerPreparation1150.png',
          stateName: 'Preparation1150'
        }
      ],
      plane: [
        {
          imgPath: 'https://hz-4.matpool.com:26142/static/ImgPool/planePreparation3141.png',
          stateName: 'Preparation3141'
        }
      ]
    },
    objName: 'owl',
    objImgPath: 'https://hz-4.matpool.com:26142/static/ImgPool/owlPreparation4737.png',
    stateName: 'Preparation4737'
  })
  const [comImgAdd, setComImgAdd] = useState('') //combinaion选择后添加的Img地址
  const [comType, setComType] = useState('')
  const [tra_data, setTraData] = useState({})
  const [inc_data, setIncData] = useState([])
  const [isInclord, setIsInclord] = useState(true) //incubation lording
  const [isPrelord, setIsPrelord] = useState(true) //preparation lording
  const [exp_data, setExpData] = useState([])
  const [exp_img, setExpImg] = useState([])
  const [exp_input_reason, setExpInputReason] = useState('')
  const [isExplord, setIsExplord] = useState(true) //exploration lording
  const [selectImgName, setImgName] = useState('') //默认值
  const [selectImgAdd, setImgAdd] = useState('') //发给后端的address

  // 设置是否完成了当前步骤，只有保存（store）或者放弃（Discard）或者preparation的create后，该状态才会变true
  const [isCompleted, setIsCompleted] = useState(true)
  const [currentState, setCurrentState] = useState(-1)

  //下半部分右上角的确定button相关请求交互
  const [isStateDone, setIsStateDone] = useState(true)

  const addTimelineNodes = (newNode) => {
    if (newNode.state === PREPARATION) {
      // preparation
      currentTimelineIndex += 1
      setCurrentTimelineIndex(currentTimelineIndex)
    }
    if (timelineData.length <= currentTimelineIndex) {
      timelineData.push({
        name: nodeType,
        nodes: [],
        links: []
      })
    }
    const currentTimeline = timelineData[currentTimelineIndex]
    if (currentTimeline.nodes.length > 0) {
      const lastNode = currentTimeline.nodes[currentTimeline.nodes.length - 1]
      if (lastNode.state === PREPARATION) {
        lastNode.score = newNode.score
      }
      currentTimeline.links.push({
        source: lastNode.id,
        target: newNode.id
      })
    }
    newNode.timelineIndex = currentTimelineIndex
    currentTimeline.nodes.push(newNode)
    timelineData[currentTimelineIndex] = currentTimeline
    setTimelineData([...timelineData])
  }

  function getTimelineEndStateNames() {
    return timelineData.map((timeline) => {
      const endNode = timeline.nodes[timeline.nodes.length - 1]
      return endNode.stateName
    })
  }

  function jumpToTimeline() {
    const index = STATES.indexOf(TIMELINE)
    handleClick(topButtons[index], index)
    setPage(
      <>
        <Timeline
          id="timeline"
          data={timelineData}
          comImgAdd={comImgAdd}
          comType={comType}
          style={{
            background: 'white'
          }}
          onTimelineClick={(index) => {
            const timeline = timelineData[index]
            const endNode = timeline.nodes[timeline.nodes.length - 1]
            const stateName = endNode.stateName
            const prevStateName =
              timeline.nodes.length > 1
                ? timeline.nodes[timeline.nodes.length - 2].stateName
                : DUMMY
            const AISketch = endNode.AISketch
            const AISketchName = endNode.AISketchName
            const timelineIndex = endNode.timelineIndex
            setImgAdd(AISketch)
            setImgName(AISketchName)
            setIncData([])
            setNowStateName(stateName)
            setCurrentTimelineIndex(timelineIndex)
            setIsCompleted(true)
          }}
        ></Timeline>
      </>
    )
  }

  useEffect(() => {
    if (STATES[menu] !== TIMELINE) {
      setCurrentState(STATES[menu])
    }

    switch (menu) {
      case 0:
        setPage(<></>)
        if (isCompleted) {
          setImgAdd('')
          // 获得第一个Preparation的stateName
          function getStateName() {
            axios.post('/api/GetStateName').then((res) => {
              const stateName = res.data.StateName
              console.log('GetStateName', stateName)
              if (!!stateName && stateName !== NONE_STATE) {
                setNowStateName(stateName)
                setIsCompleted(false)
              }
            })
          }
          var timeID = window.setTimeout(getStateName, 1000)
        }

        break
      case 1:
        setPage(
          <>
            <Incubation step={step} setStep={setStep} inc_data={inc_data} isInclord={isInclord} />
          </>
        )
        setIsCompleted(false)
        break
      case 2:
        setPage(
          <>
            <InsCombination
              setMenu={setMenu}
              data={com_data}
              setComImgAdd={setComImgAdd}
              setComType={setComType}
              selectImgAdd={selectImgAdd}
            />
          </>
        )
        setIsCompleted(false)
        break
      case 3:
        setPage(
          <>
            <InsExploration
              setMenu={setMenu}
              exp_data={exp_data}
              setExpImg={setExpImg}
              setSexp={setSexp}
              exp_input_reason={exp_input_reason}
              setExpInputReason={setExpInputReason}
              isExplord={isExplord}
            />
          </>
        )
        setIsCompleted(false)
        break
      case 4:
        setPage(
          <>
            <InsTransformation setMenu={setMenu} setStra={setStra} setTraData={setTraData} />
          </>
        )
        setIsCompleted(false)
        break
      case 5: // timeline
        jumpToTimeline()
        break
      case 6: // End
        setPage(
          <Timeline
            id="timeline"
            data={timelineData}
            comImgAdd={comImgAdd}
            comType={comType}
            style={{
              background: 'white'
            }}
            onTimelineClick={(index) => {
              const timeline = timelineData[index]
              const endNode = timeline.nodes[timeline.nodes.length - 1]
              const stateName = endNode.stateName
              const prevStateName =
                timeline.nodes.length > 1
                  ? timeline.nodes[timeline.nodes.length - 2].stateName
                  : DUMMY
              const AISketch = endNode.AISketch
              const AISketchName = endNode.AISketchName
              const timelineIndex = endNode.timelineIndex
              setImgAdd(AISketch)
              setImgName(AISketchName)
              setIncData([])
              setNowStateName(stateName)
              setCurrentTimelineIndex(timelineIndex)
              setIsCompleted(true)
            }}
          ></Timeline>
        )
        break
      default:
        break
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    exp_data,
    exp_input_reason,
    inc_data,
    com_data,
    isCompleted,
    menu,
    step,
    isInclord,
    isExplord,
    comImgAdd
  ])

  const topButtons = [
    { src: pre },
    { src: incubation },
    { src: ins_combination },
    { src: ins_exploration },
    { src: ins_transformation },
    { src: timeline },
    { src: finish }
  ]

  // useEffect(() => {
  //   if (sketchingImage) {
  //     // add timeline node
  //   }
  // }, [sketchingImage])

  useEffect(() => {
    console.log('StateNameChanged: ', nowStateName)
    setIsPrelord(false)
    // if (nowStateName === undefined || nowStateName !== NONE_STATE) {
    //   setIsCompleted(false)
    //   requestNewState()
    // } else {
    //   setIsCompleted(true)
    // }
  }, [nowStateName, isPrelord])

  function requestNewState() {
    axios.post('/api/newState', { stateType: PREPARATION, prevStateName: DUMMY }).then((res) => {
      const stateName = res.data.stateName
      if (!!stateName) {
        setImageSteps(0)
        setIncData([])
        setIsCompleted(false)
        console.log('newState', stateName)
        setNowStateName(stateName)
      } else {
        async function asyncRequestNewState() {
          await new Promise((r) => setTimeout(r, 5000))
          console.log('requestNewState', stateName)
          requestNewState()
        }
        asyncRequestNewState()
      }
    })
  }

  async function createNewNode(AISketch, AISketchName) {
    funClick(AISketchName, () => {
      // create a new node
      const node = {
        id: new Date().toISOString()
      }
      node.score = score
      node.state = currentState
      node.stateName = nowStateName
      node.AISketch = AISketch
      node.AISketchName = AISketchName
      addTimelineNodes(node)
      setMenu(STATES.indexOf(TIMELINE))
      jumpToTimeline()
      setIsCompleted(true)
    })
  }

  function handleClick(topButton, index) {
    if (newStateFlag !== index) {
      if (STATES[index] === PREPARATION && isCompleted) {
        requestNewState()
      } else if (STATES[index] === INCUBATION || STATES[index] === EXPLORATION) {
        const state = { stateType: STATES[index], prevStateName: nowStateName }
        axios.post('/api/newState', state).then((res) => {
          console.log(`${STATES[index]} newState`, res.data.stateName)
          setNowStateName(res.data.stateName)
          if (STATES[index] === EXPLORATION) {
            setIsExplord(false)
            // console.log('set res', res.data)
            setExpData(res.data)
          } else if (STATES[index] === INCUBATION) {
            setIsInclord(false)
            setIncData(res.data.sketchList)
            const len = res.data.sketchList?.length
            // console.log('len:', len)
            setStep(len - 1)
            setFirstStep(0)
            setLastStep(len - 1)
          }
        })
      } else if (STATES[index] === COMBINATION) {
        axios
          .post('/api/newState', {
            stateType: STATES[index],
            prevStateName: nowStateName,
            endStateNameList: getTimelineEndStateNames() //每条line最新节点的state name
          })
          .then((res) => {
            // debugger
            setComData(res.data)
            console.log(JSON.stringify(res.data))
            setNowStateName(res.data.stateName)
          })
      } else if (STATES[index] === TRANSFORMATION) {
        axios
          .post('/api/newState', {
            stateType: STATES[index],
            prevStateName: nowStateName
          })
          .then((res) => {
            setNowStateName(res.data.stateName)
          })
      }
      setNewStateFlag(index)
    }
  }

  function funClick(selectImgName, callback = () => {}) {
    setIsStateDone(false)
    const selectSketch = selectImgName ? selectImgName : 'owlPreparation258' // ! a trick, it really needs an actually existing img
    console.log('doneState selectSketch:', selectSketch)
    axios.post('/api/doneState', { selectSketch }).then((res) => {
      console.log('doneState', res)
      callback()
      setIsStateDone(true)
      setStra(0)
      setSexp(0)
      setExpInputReason('')
      setStoreTooltipVisible(false)
      setIsExplord(true)
      setIsInclord(true)
      setIsExplord(true)
    })
  }

  function FinishClick() {
    setMenu(6)
  }

  return (
    <div className="container">
      <div className="top">
        <h1 id="StorySketch-label">NaSketch</h1>
        {topButtons.map((topButton, index) => (
          <div
            key={index}
            className={`top-box${menu === index ? ' top-box-selected' : ''}`}
            onClick={() => {
              console.log(STATES[index], currentState)
              if (isCompleted || STATES[index] === currentState) {
                setMenu(index)
                handleClick(topButton, index)
              }
            }}
            style={{
              opacity: isCompleted || STATES[index] === currentState ? 1 : 0.3,
              width: 32 + (index === menu ? STATES[index].length * 8 : 0),
              transitionDuration: '0.5s'
            }}
          >
            <img src={topButton.src} alt="" />
            <div
              className="label1"
              style={{
                visibility: index === menu ? 'visible' : 'hidden',
                opacity: index === menu ? 1 : 0,
                transition: index === menu ? 'visibility 0.5s linear 0.5s, opacity 1s linear' : '0s'
              }}
            >
              {STATES[index]}
            </div>
          </div>
        ))}

        <Button
          disabled={isCompleted || STATES[menu] === FINISH}
          icon={<DeleteOutlined />}
          className="top-box store-btn"
          shape="circle"
          style={{
            opacity:
              STATES[menu] === TIMELINE || STATES[menu] === PREPARATION || STATES[menu] === FINISH
                ? 0
                : 1,
            color: isCompleted || STATES[menu] === TIMELINE ? 'unset' : '#FF4D4F'
          }}
          onClick={() => {
            funClick(selectImgAdd, () => {
              setMenu(STATES.indexOf(TIMELINE))
              jumpToTimeline()
              setIsCompleted(true)
            })
          }}
        >
          {/* Discard */}
        </Button>
        <Popover
          placement="bottomRight"
          visible={
            storeTooltipVisible &&
            !(isCompleted || STATES[menu] === TIMELINE || STATES[menu] === PREPARATION)
          }
          onVisibleChange={(newVisible) => {
            setStoreTooltipVisible(newVisible)
          }}
          content={
            <div
              style={{
                fontFamily: 'Inder'
              }}
            >
              <p style={{ marginBottom: 0 }}>Score it!</p>
              <Slider
                style={{ width: '50%', display: 'inline-block', verticalAlign: 'top' }}
                value={score}
                onChange={(value) => {
                  setScore(value)
                }}
              />
              <InputNumber
                min={0}
                max={100}
                style={{ width: '40%', display: 'inline-block', marginBottom: 10 }}
                value={score}
                onChange={(value) => {
                  setScore(value)
                }}
              ></InputNumber>
              <br></br>
              <Button
                loading={!isStateDone}
                type="primary"
                className="btn"
                disabled={0 <= score && score <= 100 ? false : true}
                style={{
                  width: '100%'
                }}
                onClick={() => {
                  console.log('node img', selectImgAdd)
                  createNewNode(selectImgAdd, selectImgName)
                }}
              >
                Confirm
              </Button>
            </div>
          }
        >
          <Button
            disabled={
              isCompleted ||
              STATES[menu] === TIMELINE ||
              STATES[menu] === PREPARATION ||
              STATES[menu] === FINISH
            }
            icon={<SaveOutlined />}
            className="top-box store-btn"
            shape="circle"
            style={{
              opacity: STATES[menu] === FINISH ? 0 : 1,
              display:
                STATES[menu] === TIMELINE || STATES[menu] === PREPARATION || STATES[menu] === FINISH
                  ? 'none'
                  : 'unset',
              color: isCompleted || STATES[menu] === TIMELINE ? 'unset' : '#1890FF'
            }}
            onClick={() => {
              setStoreTooltipVisible(true)
            }}
          >
            {/* Store */}
          </Button>
        </Popover>
      </div>
      <div className="board">
        <div className="board-top">
          {STATES[menu] === PREPARATION ? (
            // if image is not choose
            <div className="board-left-1">
              <ChooseBox
                // const firstStep = d3.min(Object.keys(images).map((step) => +step))
                // const lastStep = d3.max(Object.keys(images).map((step) => +step))
                isPrelord={isPrelord}
                setImageSteps={setImageSteps}
                isStateDone={isStateDone}
                setNodeType={setNodeType}
                onCreate={(selectImgAdd, selectImgName) => {
                  setImgAdd(selectImgAdd)
                  setImgName(selectImgName)
                  console.log('Preparation, selectImgName:', selectImgName)
                  createNewNode(selectImgAdd, selectImgName)
                }}
              ></ChooseBox>
              {/* <div className="showimg-area">
                <div className="label">Image Preview</div>
                <div className="imgpreview">
                  {chooseImgAdd == '' ? null : (
                    <img className="chooseimg" src={chooseImgAdd} alt="chooseimg" />
                  )}
                </div>
              </div> */}
            </div>
          ) : null}
          {STATES[menu] !== (FINISH || PREPARATION) ? (
            <div className="board-left-2">
              <AIDisplay
                step={step}
                setStep={(step) => {
                  setStep(Math.max(firstStep, Math.min(lastStep, step)))
                }}
                state={STATES[menu]}
                tra_data={tra_data}
                inc_data={inc_data}
                s_exp={s_exp}
                s_tra={s_tra}
                exp_img={exp_img}
                setImgName={setImgName}
                selectImgAdd={selectImgAdd}
                setImgAdd={setImgAdd}
                comImgAdd={comImgAdd}
              ></AIDisplay>
            </div>
          ) : null}
          {menu !== 0 || imageSteps !== 0 ? (
            <div className={`board-right${menu === 6 ? ' board-right-end' : ''}`}>
              <Canvas image={undefined} />{' '}
            </div>
          ) : null}
        </div>
        {menu !== 0 ? <div className="board-bottom">{page}</div> : null}
      </div>
      {/* <Foot setMenu={setMenu} menu={menu} /> */}
    </div>
  )
}

export default Container
