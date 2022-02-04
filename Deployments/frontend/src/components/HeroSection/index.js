import {React, useState} from 'react'
import {
    HeroContainer,
    HeroBg,
    HeroContent,
    HeroH1,
    HeroP,
    HeroBtnWrapper,
    ArrowFoward,
    ArrowRight
} from './HeroElements';
import { Button } from '../ButtonElements'


const HeroSection = () => {

    const [hover, setHover] = useState(false)

    const onHover = () =>{
        setHover(!hover)
    }

    return (
        <HeroContainer id = "home">
            <HeroBg>
                               
            </HeroBg>
            <HeroContent>
                <HeroH1>
                    SSOC Autcoder
                </HeroH1>
                <HeroP>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
                    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                </HeroP>
                <HeroBtnWrapper>
                    <Button to="search" 
                    onMouseEnter = {onHover} 
                    onMouseLeave = {onHover}
                    primary = "true"
                    >
                        Get Started {hover ? <ArrowFoward /> : <ArrowRight />}
                    </Button>
                </HeroBtnWrapper>
            </HeroContent>
        </HeroContainer>
    )
}

export default HeroSection
