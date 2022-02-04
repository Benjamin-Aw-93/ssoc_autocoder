import styled from "styled-components";
import { MdKeyboardArrowRight, MdArrowForward } from 'react-icons/md';


export const HeroContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 30px;
  height: 800px;
  position: relative;
  z-index: 1;
`


export const HeroBg = styled.div`
  background: #d9d9d9;
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 50%;
  height: 100%;
  overflow: hidden;
`

export const HeroContent = styled.div`
  z-index: 3;
  left: 0;
  width: 50%;
  position: absolute;
  padding: 8px 120px;
  display: flex;
  flex-direction: column;
  align-items: left;

  @media screen and (max-width: 768px){
    padding: 8px 30px;
  }

  @media screen and (max-width: 480px){
    padding: 8px 24px;
  }
`

export const HeroH1 = styled.h1`
  color: #000;
  font-size:48px;
  text-align: left;

  @media screen and (max-width: 768px){
      font-size:40px;
  }

  @media screen and (max-width: 480px){
      font-size:32px;
  }
`

export const HeroP = styled.p`
  margin-top: 24px;
  margin-bottom: 24px;
  color: #000;
  font-size:24px;
  text-align: left;
  max-width: 600px;

  @media screen and (max-width: 768px){
      font-size:18px;
  }

  @media screen and (max-width: 480px){
      font-size:16px;
  }
`

export const HeroBtnWrapper = styled.div`
  display:flex;
  margin-top: 32px;
  flex-direction: row;
  align-items: center;
`

export const ArrowFoward = styled(MdArrowForward)`
  margin-left: 8px;
  font-size: 20px;
`

export const ArrowRight = styled(MdKeyboardArrowRight)`

`