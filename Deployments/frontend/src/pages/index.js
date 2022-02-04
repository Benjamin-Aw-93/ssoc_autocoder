import React, {useState, useEffect} from 'react'
import HeroSection from '../components/HeroSection';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import SearchSection from '../components/SearchSection';
import getSSOCData from '../components/API/lambdaAPI';
import ResultsSection from '../components/ResultsSection';
import { makeStyles } from '@material-ui/core';

const Home = () => {

    const useStyles = makeStyles(theme => ({
        root: {
          borderRadius: 12,
          minWidth: 256,
          textAlign: 'center',
        },
        header: {
          textAlign: 'center',
          spacing: 10,
        },
        list: {
          padding: '20px',
        },
        button: {
          margin: theme.spacing(1),
        },
        action: {
          display: 'flex',
          justifyContent: 'space-around',
        },
        accordianHeading: {
          fontSize: theme.typography.pxToRem(15),
          flexBasis: '33.33%',
          flexShrink: 0, 
          textAlign: 'center',
          flexGrow: 0,
        },
        accordianSecondaryHeading: {
          fontSize: theme.typography.pxToRem(15),
          color: theme.palette.text.secondary,
          textAlign: 'center',
        },
      }));

    const [isOpen, setIsOpen] = useState(false);
    const [mcfID, setMcfId] = useState("");
    const [isPress, setisPress] = useState(false);
    const [mainResult, setmainResult] = useState({});

    const toggle = () => {
        setIsOpen(!isOpen)
    }

    const togglePress = () => {
        setisPress(!isPress)
    }

    console.log(mainResult)

    useEffect(() => {
        getSSOCData(`${mcfID}`)
          .then(data => {
            setmainResult({
                ...mainResult,
                ...data.data
            })
          });
        
        return () => {
            console.log("Component unmounted")
        };
      }, [isPress]);


    return (
        <>
            <Sidebar isOpen = {isOpen} toggle = {toggle}/>
            <Navbar toggle = {toggle}/>
            <HeroSection/>
            <SearchSection setMcfId = {setMcfId} mcfID = {mcfID} togglePress = {togglePress}/>
            <ResultsSection useStyles = {useStyles} mainResult = {mainResult} ></ResultsSection>
        </>
    )
}

export default Home
