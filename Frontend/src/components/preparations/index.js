import React, { useEffect, useState } from 'react'
import { Button, Input, Tag, Drawer, Spin } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'
import 'antd/dist/antd.css'
import './index.scss'
import axios from 'axios'

const { Search } = Input

const antIcon = (
  <LoadingOutlined
    style={{
      fontSize: 24
    }}
    spin
  />
)

const ImgChosoe = ({ imgId, setImgId, data, onConfirm, isStateDone }) => {
  const [chosedImage, setChosedImage] = useState(undefined)
  const [chosedImageName, setChosedImageName] = useState(undefined)
  const [drawerVisible, setDrawerVisible] = useState(false)
  useEffect(() => {
    if (chosedImage) {
      setDrawerVisible(true)
    }
  }, [chosedImage])
  return (
    <div
      className="site-drawer-render-in-current-wrapper"
      style={{
        display: 'flex',
        width: '100%',
        height: '100%'
      }}
    >
      <div className="img_board">
        {data.map((info, index) => (
          <img
            className={`img${imgId === index ? ' chosen' : ''}`}
            onClick={() => {
              setImgId(index)
              setChosedImageName(info['imgName'])
              setChosedImage(info['imgPath'])
            }}
            src={info['imgPath']}
            key={index}
            alt={index}
          />
        ))}
      </div>
      <Drawer
        placement="right"
        closable={false}
        onClose={() => {
          setDrawerVisible(false)
        }}
        visible={drawerVisible}
        getContainer={false}
        style={{ position: 'absolute' }}
        bodyStyle={{ padding: 20 }}
      >
        <div className="imgpreview">
          <img className="chooseimg" src={chosedImage} alt="chooseimg" />

          <div className="button">
            <Button
              loading={!isStateDone}
              type="primary"
              onClick={() => {
                onConfirm(chosedImage, chosedImageName)
              }}
            >
              Create
            </Button>
          </div>
        </div>
      </Drawer>
      {/* {chosedImage ? (
        <div className="imgpreview">
          <img className="chooseimg" src={chosedImage} alt="chooseimg" />

          <div className="button">
            <Button
              type="primary"
              // onClick={async () => {
              //   // TODO: fetch a real AI image from server
              //   // const images = await fetchAIImage(imgId)
              //   // chooseImages(images)
              //   handleClick(img_data, imgId)
              // }}
              onClick={onConfirm}
            >
              Create
            </Button>
          </div>
        </div>
      ) : null} */}
    </div>
  )
}

const ChooseBox = ({ setImageSteps, isStateDone, onCreate, isPrelord, setNodeType }) => {
  const [imgId, setImgId] = useState(-1)
  const [img_data, setImgData] = useState([]) //response中的Img数组
  const [islording, setLording] = useState(false) //图片处是否显示“加载中”
  const [searchWord, setSearchWord] = useState('')

  const onSearch = (value) => {
    setSearchWord(value)
    setNodeType(value)
    setLording(true)
    axios.post('/api/PrepGetObjs', { objName: value }).then((res) => {
      const temp_data = res.data.sketchList
      // console.log('temp_data type: ', typeof temp_data)
      const test_data = []
      temp_data.forEach((element) => {
        test_data.push({ imgPath: element.imgPath, imgName: element.sketchName })
      })
      // console.log(test_data)
      // console.log(img_data)
      // setImgData({ imgPath: test_data.imgPath, imgName: test_data.imgName })
      setImgData(test_data)
      setLording(false)
    })
  }

  const words = ['owl', 'flower', 'astronaut', 'plane', 'alien', 'rocket', 'spaceship', 'mars']

  return (
    <div
      className="choose-box"
      style={
        img_data.length
          ? {}
          : {
              left: '50%',
              top: '45%',
              transform: 'translate(-50%, -50%)'
            }
      }
    >
      <Spin indicator={antIcon} tip="lording" spinning={isPrelord} />
      <Search
        className="input"
        placeholder="Input a type and press Enter"
        onSearch={onSearch}
        allowClear
        enterButton
        loading={islording}
        value={searchWord}
      />
      <div className="button-choose-area">
        {words.map((word) => (
          <Tag
            className="button-choose"
            key={word}
            onClick={() => {
              onSearch(word)
            }}
          >
            {word}
          </Tag>
        ))}
      </div>

      {img_data.length ? (
        <div className="preparation-img">
          <ImgChosoe
            imgId={imgId}
            setImgId={setImgId}
            data={img_data}
            isStateDone={isStateDone}
            onConfirm={(chosedImage, chosedImageName) => {
              // TODO: fetch a real AI image from server
              console.log('preparation', chosedImage)
              setImageSteps(1)
              onCreate(chosedImage, chosedImageName)
            }}
          />
        </div>
      ) : null}
    </div>
  )
}

export { ChooseBox }
