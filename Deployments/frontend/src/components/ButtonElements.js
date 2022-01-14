import styled from 'styled-components'
import {Link} from 'react-scroll'

export const Button = styled(Link)`
    border-radius: ${({squared}) => (squared ? '10px' : '50px')};
    background: ${({primary}) => (primary ? '#ffb74d' : '#fff')};
    white-space: nowrap;
    padding: ${({big}) => (big ? '14px 48px': '12px 30px')};
    color: ${(primary) => (primary ? '#000' : '#fff')};
    font-size: ${({fontBig}) => (fontBig ? '20px' : '16px')};
    outline: none;
    border: 2px solid #ffb74d;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.2s ease-in-out;

    &:hover{
        transition: all 0.2s ease-in-out;
        background: ${({primary}) => (primary ? '#fdd14d' : '#fdd14d')};
        border: none;
    }

`