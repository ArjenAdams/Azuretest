import Button from "../../ui/Button";

export default function ButtonTest() {
    return (
        <div className="m-8">
            <h1>Test Buttons</h1>
            <Button onClick={()=>{alert("Clicked primary")}}  >
                Primary md
            </Button>

            <Button secondary size="xl">
                Secondary xl
            </Button>

            <Button size="sm">
                Primary sm
            </Button>
        </div>
    )
}