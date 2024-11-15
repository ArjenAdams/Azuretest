import clsx from "clsx";

type ButtonProperties = {
    size?       : "xl" | "md" | "sm",
    primary?    : boolean,
    secondary?  : boolean,
    children?    : string,
    onClick?     : () => void
}

function determineSize(size : string) {
    // Default size is md, could remove md option but some people might like it for semantics
    if(!size){
        return "px-3 py-1 text-md";
    }
    if(size === "xl"){
        return "px-6 py-3 text-xl";
    }else if(size === "sm"){ 
        return "px-2 py-1 text-sm";
    }else{
        return "px-4 py-2 text-md";
    }
}

function determineColor(secondary : boolean){
    // Default color is primary, could remove primary option but some people might like it for semantics
    if(secondary){
        return "bg-secondary hover:bg-secondary-800 active:bg-secondary-700 transition-colors text-white duration-150";
    }else{
        return "bg-primary hover:bg-primary-500 active:bg-primary-600 transition-colors text-black duration-150"
    }

}

export default function Button(properties : ButtonProperties){
    const sizeString = determineSize(properties.size || "");
    // If secondary is not passed it means it is false
    const colorString = determineColor(properties.secondary || false);

    return (
        <button 
            // If there is no onClick passed we just pass an empty void function
            onClick={properties.onClick? properties.onClick : ()=>{}} 
            className={clsx("rounded-[40px]", sizeString, colorString)}
        >
            {properties.children}           
        </button>
    )
}