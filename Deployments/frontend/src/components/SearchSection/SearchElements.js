import styled from 'styled-components'

export const SearchContainer = styled.div`
  color: #fff;
  background: #f2f2f2;

  @media screen and (max-width: 768px) {
      padding: 100px 0;
  }
`

export const SearchWrapper = styled.div`
  display: grid;
  z-index: 1;
  height: 860px;
  width: 80%;
  max-width: 1100px;
  margin-right: auto;
  margin-left: auto;
  padding: 0 24px;
  justify-content: center;
`

export const SearchRow = styled.div`
  display: flex;
  flex-direction: column;
  z-index: 0;
  padding: 320px 0px;
`

export const TextWrapper = styled.div`
   max-width: 540px;
`

export const SearchBarWrapper = styled.div`
  max-width: 540px;
`

export const Heading = styled.h1`
  margin-bottom: 24px;
  font-size: 48px;
  line-height: 1.1;
  font-weight: 600;
  color: #000;
  
  @media screen and (max-width: 768px) {
      font-size: 32px;
  }
`

export const Subtitle = styled.p`
  max-width:440px;
  margin-bottom: 35px;
  font-size: 18px;
  line-height: 24px;
  color: #000;
`

export const BtnWrap = styled.div`
  display: grid;
  padding-top: 20px;
  grid-template-columns: 1fr 1fr;
  grid-gap: 20px;
  justify-content: flex;
`

