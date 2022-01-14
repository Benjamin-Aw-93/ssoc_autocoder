import React from 'react'
import {FaBars} from 'react-icons/fa'
import {
    Nav, 
    NavbarContainer, 
    NavLogo, 
    MobileIcon, 
    NavMenu, 
    NavItem, 
    NavLinks,
    NavBtn,
    NavBtnLink
} from './NavbarElements';


const Navbar = ({ toggle }) => {
    return (
        <>
            <Nav>
                <NavbarContainer>
                    <NavLogo to = '/'>SSOC AUTOCODER</NavLogo>
                    <MobileIcon onClick = {toggle}>
                        <FaBars />
                    </MobileIcon>
                    <NavMenu>
                        <NavItem>
                            <NavLinks to ="search">Search</NavLinks>
                        </NavItem>
                        <NavItem>
                            <NavLinks to ="about">About</NavLinks>
                        </NavItem>
                        <NavItem>
                            <NavLinks to ="api">API</NavLinks>
                        </NavItem>
                        <NavItem>
                            <NavLinks to ="contact-us">Contact Us</NavLinks>
                        </NavItem>
                    </NavMenu>
                    <NavBtn>
                        <NavBtnLink to="/load-api">Load API Key</NavBtnLink>
                    </NavBtn>
                </NavbarContainer>
            </Nav>
        </>
    )
}

export default Navbar
