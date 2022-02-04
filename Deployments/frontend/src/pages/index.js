/* Importing packages */
import React, {useState, useEffect, useRef} from 'react'
import SearchSection from '../components/SearchSection';
import getSSOCData from '../components/API/lambdaAPI';
import ResultsSection from '../components/ResultsSection';
import LoadingSection from '../components/LoadingSection';
import { AlertTitle } from '@mui/material';
import { Alert } from '@mui/material';
import { Snackbar } from '@mui/material';

/* This page acts as the home page for all the three components, the search, the loading screen and the results page*/
/* Almost all of the components logic and states are controlled through here*/
const Home = () => {

  /*useRef hook to locate where the compoenet is, for scrolling purposes, does not tigger component update when value is updated*/
  const loadingScreenRef = useRef(null);
  const searchBarRef = useRef(null);

    function isEmpty(obj) {
      return Object.keys(obj).length === 0;
    }
    /* Controlling scroll functionality */
    const executeScroll = () => loadingScreenRef.current.scrollIntoView({  behavior: 'smooth' })
    const executeSearchScroll = () => {
      searchBarRef.current.focus();
      setMcfUrl('')
      setmainResult({});
    }
    /* All states required */
    const [mcfUrl, setMcfUrl] = useState(""); /* User Input*/
    const [mainResult, setmainResult] = useState({}); /* Result */
    const [isLoading, setisLoading] = useState(false); /* When the app is loading show this page */
    const [isError, setisError] = useState(false); /* If an error is returned by the API call */
    const [errorMessage, seterrorMessage] = useState({}); /* What the error message returned by the API call is */

    
    useEffect(() => {
      executeScroll();
    }, [isLoading]);

    /* When search button is pressed, what happens accordingly */
    const togglePress = () => {
        setmainResult({});
        setisLoading(true);
    
        getSSOCData(`${mcfUrl}`)
        .then(data => {
          setmainResult({
              ...mainResult,
              ...data.data
          });
        })
        .catch(e => {
          seterrorMessage({
            ...errorMessage,
            ...e.response
          });
          setisError(true);
        })
        .then(() => {
          setisLoading(false);
        })
    }

    /* When reset button is pressed, what happens accordingly */
    const toggleDefault = () => {
      setmainResult({});
      setisLoading(true);
  
      getSSOCData(`feelinglucky`)
      .then(data => {
        setmainResult({
            ...mainResult,
            ...data.data
        });
      })
      .catch(e => {
        seterrorMessage({
          ...errorMessage,
          ...e.response
        });
        setisError(true);
      })
      .then(() => {
        setisLoading(false);
      })
  }
    /*Closing error button*/
    const handleSnackClose = (event, reason) => {
      if (reason === 'clickaway') {
        return;
      }
  
      setisError(false);
    };

    return (
        <>
            <SearchSection searchBarRef = {searchBarRef} setMcfUrl = {setMcfUrl} mcfUrl = {mcfUrl} togglePress = {togglePress} toggleDefault = {toggleDefault}/>
            <div ref={loadingScreenRef}>
              {isLoading ? <LoadingSection></LoadingSection> : null}
            </div>
            {isEmpty(mainResult) ? null : (<ResultsSection executeSearchScroll = {executeSearchScroll} mainResult = {mainResult} ></ResultsSection>) }
            {isEmpty(errorMessage)? null: (
            <Snackbar open={isError} autoHideDuration={6000} onClose={handleSnackClose}>
              <Alert onClose={handleSnackClose} severity="error" sx={{ width: '100%' }}>
                <AlertTitle><strong>Error {errorMessage.status}!</strong></AlertTitle>
                {errorMessage.data.detail}
              </Alert>
            </Snackbar>)}
        </>
    )
}

export default Home